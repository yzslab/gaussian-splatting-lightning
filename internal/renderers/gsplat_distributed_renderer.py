from dataclasses import dataclass
from gsplat import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import spherical_harmonics
from .renderer import *
import torch.distributed.nn.functional

DEFAULT_BLOCK_SIZE: int = 16
DEFAULT_ANTI_ALIASED_STATUS: bool = True


@dataclass
class MemberData:
    # camera

    width: int
    height: int

    fx: float
    fy: float
    cx: float
    cy: float

    camera_center: torch.Tensor

    w2c: torch.Tensor

    # gaussians
    n_gaussians: int


@dataclass
class GSplatDistributedRenderer(RendererConfig):
    block_size: int = DEFAULT_BLOCK_SIZE

    anti_aliased: bool = DEFAULT_ANTI_ALIASED_STATUS

    rebalance_interval: int = -1

    def instantiate(self, *args, **kwargs) -> Renderer:
        return GSplatDistributedRendererImpl(self)


class GSplatDistributedRendererImpl(Renderer):
    # TODO: the metrics of Lego scene are a little bit lower than non-distributed version, and the number of Gaussians is only about half.

    def __init__(self, config: GSplatDistributedRenderer) -> None:
        super().__init__()

        self.block_size = config.block_size
        self.anti_aliased = config.anti_aliased
        self.rebalance_interval = config.rebalance_interval

        self.world_size = 1
        self.global_rank = 0

    def training_setup(self, module: lightning.LightningModule) -> Tuple[
        Optional[Union[
            List[torch.optim.Optimizer],
            torch.optim.Optimizer,
        ]],
        Optional[Union[
            List[torch.optim.lr_scheduler.LRScheduler],
            torch.optim.lr_scheduler.LRScheduler,
        ]]
    ]:
        self.world_size = module.trainer.world_size
        self.global_rank = module.trainer.global_rank

        # divide gaussians evenly
        n_gaussians = module.gaussian_model.get_xyz.shape[0]
        n_gaussians_per_member = round(n_gaussians / self.world_size)

        l = n_gaussians_per_member * self.global_rank
        r = l + n_gaussians_per_member
        if self.global_rank + 1 == self.world_size:
            r = n_gaussians

        new_param_tensors = {}
        for attr_name, param_group_name in module.gaussian_model.param_names:
            new_param_tensors[param_group_name] = getattr(module.gaussian_model, attr_name)[l:r]

        self.replace_tensors_to_optimizer(new_param_tensors, module.gaussian_model)
        module.gaussian_model.xyz_gradient_accum = module.gaussian_model.xyz_gradient_accum[l:r]
        module.gaussian_model.denom = module.gaussian_model.denom[l:r]
        module.gaussian_model.max_radii2D = module.gaussian_model.max_radii2D[l:r]

        print(f"rank={self.global_rank}, l={l}, r={r}")

        return None, None

    @staticmethod
    def replace_tensors_to_optimizer(tensors_dict, gaussian_model, inds=None):
        # # get current parameters
        # tensors_dict = {}
        # for attr_name, param_group_name in gaussian_model.param_names:
        #     tensors_dict[param_group_name] = getattr(gaussian_model, attr_name)

        optimizable_tensors = {}
        for group in gaussian_model.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = gaussian_model.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                if inds is not None:
                    stored_state["exp_avg"][inds] = 0
                    stored_state["exp_avg_sq"][inds] = 0
                else:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                # replace with new tensor and state
                del gaussian_model.optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter(tensor.requires_grad_(True))
                gaussian_model.optimizer.state[group['params'][0]] = stored_state
            else:
                group["params"][0] = torch.nn.Parameter(tensor.requires_grad_(True))

            optimizable_tensors[group["name"]] = group["params"][0]

        # update
        for attr_name, param_group_name in gaussian_model.param_names:
            setattr(gaussian_model, attr_name, optimizable_tensors[param_group_name])

    def gather_member_data(self, viewpoint_camera: Camera, n_gaussians: int, device) -> List[MemberData]:
        # gather data from group members
        ## create local float tensor
        float_tensor = torch.tensor([
            viewpoint_camera.fx,
            viewpoint_camera.fy,
            viewpoint_camera.cx,
            viewpoint_camera.cy,
        ], dtype=torch.float, device=device)
        float_tensor = torch.concat([float_tensor, viewpoint_camera.camera_center, viewpoint_camera.world_to_camera.reshape((-1))], dim=-1)
        ## perform float tensor gathering
        gathered_float_tensors = torch.distributed.nn.functional.all_gather(float_tensor)

        ## create local int tensor
        int_tensor = torch.tensor([
            viewpoint_camera.height.int(),
            viewpoint_camera.width.int(),
            n_gaussians,
        ], dtype=torch.int, device=device)
        ## perform int tensor gathering
        gathered_int_tensors = torch.distributed.nn.functional.all_gather(int_tensor)

        # reformat gathered data
        member_data_list = []
        for i in range(len(gathered_float_tensors)):
            float_tensor = gathered_float_tensors[i]
            int_tensor = gathered_int_tensors[i]
            member_data_list.append(MemberData(
                width=int_tensor[0].item(),
                height=int_tensor[1].item(),
                fx=float_tensor[0].item(),
                fy=float_tensor[1].item(),
                cx=float_tensor[2].item(),
                cy=float_tensor[3].item(),
                camera_center=float_tensor[4:7],
                w2c=float_tensor[7:].reshape((4, 4)),
                n_gaussians=int_tensor[2].item(),
            ))

        return member_data_list

    def rasterizer_required_data_all2all(
            self,
            projection_result_list: List[Tuple],
            rgb_list: List[torch.Tensor],
            opacities: torch.Tensor,
            member_data_list: List[MemberData],
            device,
    ) -> Tuple[List, torch.Tensor, torch.Tensor]:
        output_float_tensor_list = []
        input_float_tensor_list = []
        output_int_tensor_list = []
        input_int_tensor_list = []

        for i in range(len(projection_result_list)):
            xys, depths, radii, conics, comp, num_tiles_hit, cov3d = projection_result_list[i]
            """
            xys: [N, 2]
            depths: [N], int
            radii: [N], int
            conics: [#, 3]
            comp: [N]
            num_tile_hit: [N], int
            cov3d: [N, 6]
            
            opacities: [N, 1]
            rgb: [N, 3]
            """

            # build float tensor sent to other members
            float_tensor = torch.concat([
                xys,
                depths.unsqueeze(-1),
                conics,
                comp.unsqueeze(-1),
                # cov3d,  # this is not required by rasterization
                opacities,
                rgb_list[i],
            ], dim=-1)
            # build int tensor
            int_tensor = torch.concat([
                radii.unsqueeze(-1),
                num_tiles_hit.unsqueeze(-1),
            ], dim=-1)
            # append to input list
            input_float_tensor_list.append(float_tensor)
            input_int_tensor_list.append(int_tensor)

            # create output tensors and append to correspond lists
            output_float_tensor_list.append(torch.empty(
                (member_data_list[i].n_gaussians, float_tensor.shape[-1]),
                dtype=torch.float,
                device=device,
            ))
            output_int_tensor_list.append(torch.empty(
                (member_data_list[i].n_gaussians, int_tensor.shape[-1]),
                dtype=torch.int,
                device=device,
            ))

        # All-to-All
        torch.distributed.nn.functional.all_to_all(
            output_tensor_list=output_float_tensor_list,
            input_tensor_list=input_float_tensor_list,
        )
        torch.distributed.nn.functional.all_to_all(
            output_tensor_list=output_int_tensor_list,
            input_tensor_list=input_int_tensor_list,
        )

        # post-processing
        float_tensor = torch.concat(output_float_tensor_list, dim=0)
        int_tensor = torch.concat(output_int_tensor_list, dim=0)

        xys, depths, conics, comp, opacities, rgbs = torch.split(
            float_tensor,
            [2, 1, 3, 1, 1, 3],
            dim=-1,
        )
        radii, num_tiles_hit = torch.split(
            int_tensor,
            [1, 1],
            dim=-1,
        )

        return [
            xys, depths.squeeze(-1), radii.squeeze(-1), conics, comp.squeeze(-1), num_tiles_hit.squeeze(-1),
        ], opacities, rgbs

    def project(self, member_data: MemberData, pc: GaussianModel, scaling_modifier):
        results = project_gaussians(
            means3d=pc.get_xyz,
            scales=pc.get_scaling,
            glob_scale=scaling_modifier,
            quats=pc.get_rotation,
            viewmat=member_data.w2c.T,
            # projmat=viewpoint_camera.full_projection.T,
            fx=member_data.fx,
            fy=member_data.fy,
            cx=member_data.cx,
            cy=member_data.cy,
            img_height=member_data.height,
            img_width=member_data.width,
            block_width=self.block_size,
        )

        try:
            results[0].retain_grad()
        except:
            pass

        return results

    def forward(self, viewpoint_camera: Camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0, render_types: list = None, **kwargs):
        # gather camera and number of gaussian from group members
        # assert all members have different camera (this is the default behavior for training set)
        member_data_list = self.gather_member_data(viewpoint_camera, pc.get_xyz.shape[0], bg_color.device)

        # perform the projection and SH for each member's camera
        projection_results_list = []
        rgb_list = []
        for member_data in member_data_list:
            # store projection results to list
            projection_results_list.append(self.project(member_data, pc, scaling_modifier))

            # store SH results to list
            viewdirs = pc.get_xyz.detach() - member_data.camera_center  # (N, 3)
            rgbs = spherical_harmonics(pc.active_sh_degree, viewdirs, pc.get_features)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
            rgb_list.append(rgbs)

        # perform All-to-All operation
        projection_results, opacities, rgbs = self.rasterizer_required_data_all2all(
            projection_result_list=projection_results_list,
            rgb_list=rgb_list,
            opacities=pc.get_opacity,
            member_data_list=member_data_list,
            device=bg_color.device,
        )

        # rasterization below is the same as non-distributed renderer

        xys, depths, radii, conics, comp, num_tiles_hit = projection_results

        if self.anti_aliased is True:
            opacities = opacities * comp[:, None]

        local_camera_data = member_data_list[self.global_rank]
        img_height = local_camera_data.height
        img_width = local_camera_data.width

        rgb = rasterize_gaussians(  # type: ignore
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,  # type: ignore
            rgbs,
            opacities,
            img_height=img_height,
            img_width=img_width,
            block_width=self.block_size,
            background=bg_color,
            return_alpha=False,
        )  # type: ignore
        rgb = rgb.permute(2, 0, 1)

        # a little difference below, since the densification needs projection results from all cameras
        return {
            "render": rgb,
            "member_data_list": member_data_list,
            "projection_results_list": projection_results_list,
            "xys_grad_scale_required": True,
        }

    def training_forward(self, step: int, module: lightning.LightningModule, viewpoint_camera: Camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0, render_types: list = None, **kwargs):
        # TODO: rebalance gaussians distribution, and need to be performed after densification or pruning
        return super().training_forward(step, module, viewpoint_camera, pc, bg_color, scaling_modifier, render_types, **kwargs)

    def get_available_outputs(self) -> Dict:
        return {
            "rgb": RendererOutputInfo("render"),
        }
