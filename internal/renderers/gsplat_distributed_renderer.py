import traceback
from dataclasses import dataclass
from gsplat import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import spherical_harmonics
from .renderer import *
from lightning.pytorch.profilers import PassThroughProfiler
from internal.density_controllers.density_controller import Utils as DensityControllerUtils
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

    appearance_id: torch.Tensor

    normalized_appearance_id: torch.Tensor

    # gaussians
    n_gaussians: int


@dataclass
class GSplatDistributedRenderer(RendererConfig):
    block_size: int = DEFAULT_BLOCK_SIZE

    anti_aliased: bool = DEFAULT_ANTI_ALIASED_STATUS

    # Since the density controllers are replaceable, below parameters should be updated manually when the parameters of density controller changed

    redistribute_interval: int = 1000
    """This value should be the result of `n` times the densify interval, where `n` is an integer"""

    redistribute_until: int = 15_000
    """Should be the same as the densify until iteration"""

    redistribute_threshold: float = 1.1
    """Redistribute if min*threshold < max"""

    def instantiate(self, *args, **kwargs) -> Renderer:
        return GSplatDistributedRendererImpl(self)


class GSplatDistributedRendererImpl(Renderer):
    # TODO: the metrics of Lego scene are a little bit lower than non-distributed version, and the number of Gaussians is only about half.
    # Real world scenes have improvements

    def __init__(self, config: GSplatDistributedRenderer) -> None:
        super().__init__()

        self.config = config

        self.block_size = config.block_size
        self.anti_aliased = config.anti_aliased

        self.world_size = 1
        self.global_rank = 0

        self.profile_prefix = "[Renderer]GSplatDistributedRenderer."
        self.profiler = PassThroughProfiler()

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
        n_gaussians = module.gaussian_model.n_gaussians
        n_gaussians_per_member = round(n_gaussians / self.world_size)

        l = n_gaussians_per_member * self.global_rank
        r = l + n_gaussians_per_member
        if self.global_rank + 1 == self.world_size:
            r = n_gaussians

        new_param_tensors = {}
        for attr_name, attr_value in module.gaussian_model.properties.items():
            new_param_tensors[attr_name] = attr_value[l:r]

        self.replace_tensors_to_optimizer(new_param_tensors, module.gaussian_model, module.gaussian_optimizers)

        # notify module
        self.on_density_changed = module.density_updated_by_renderer
        self.on_density_changed()

        try:
            self.profiler = module.trainer.profiler
        except:
            traceback.print_exc()
            pass

        print(f"rank={self.global_rank}, l={l}, r={r}")

        return None, None

    @staticmethod
    def replace_tensors_to_optimizer(tensors_dict, gaussian_model, optimizers):
        gaussian_model.properties = DensityControllerUtils.replace_tensors_to_optimizers(
            tensors_dict,
            optimizers,
        )

    def gather_member_data(self, viewpoint_camera: Camera, n_gaussians: int, device) -> List[MemberData]:
        # gather data from group members
        ## create local float tensor
        float_tensor = torch.tensor([
            viewpoint_camera.fx,
            viewpoint_camera.fy,
            viewpoint_camera.cx,
            viewpoint_camera.cy,
            viewpoint_camera.normalized_appearance_id,
        ], dtype=torch.float, device=device)
        float_tensor = torch.concat([float_tensor, viewpoint_camera.camera_center, viewpoint_camera.world_to_camera.reshape((-1))], dim=-1)
        ## perform float tensor gathering
        gathered_float_tensors = torch.distributed.nn.functional.all_gather(float_tensor)

        ## create local int tensor
        int_tensor = torch.tensor([
            viewpoint_camera.width.int(),
            viewpoint_camera.height.int(),
            n_gaussians,
            viewpoint_camera.appearance_id,
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
                camera_center=float_tensor[5:8],
                w2c=float_tensor[8:].reshape((4, 4)),
                appearance_id=int_tensor[3],
                normalized_appearance_id=float_tensor[4],
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
        # TODO: transfer visible Gaussians only

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

        return results

    def forward(self, viewpoint_camera: Camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0, render_types: list = None, **kwargs):
        with self.profiler.profile(f"{self.profile_prefix}forward"):
            with self.profiler.profile(f"{self.profile_prefix}gather_member_data"):
                # gather camera and number of gaussian from group members
                # assert all members have different camera (this is the default behavior for training set)
                member_data_list = self.gather_member_data(viewpoint_camera, pc.get_xyz.shape[0], bg_color.device)

            with self.profiler.profile(f"{self.profile_prefix}project"):
                # perform the projection and SH for each member's camera
                projection_results_list = []
                rgb_list = []
                for member_data in member_data_list:
                    # store projection results to list
                    project_results = self.project(member_data, pc, scaling_modifier)
                    projection_results_list.append(project_results)

                    rgb_list.append(self.get_rgbs(pc, member_data, project_results))

            with self.profiler.profile(f"{self.profile_prefix}rasterizer_required_data_all2all"):
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

            with self.profiler.profile(f"{self.profile_prefix}rasterize"):
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

    def get_rgbs(self, pc, member_data: MemberData, project_results) -> torch.Tensor:
        # store SH results to list
        viewdirs = pc.get_xyz.detach() - member_data.camera_center  # (N, 3)
        rgbs = spherical_harmonics(pc.active_sh_degree, viewdirs, pc.get_features)
        rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        return rgbs

    def after_training_step(self, step: int, module):
        if self.config.redistribute_interval < 0:
            return
        if step >= self.config.redistribute_until:
            return
        if step % self.config.redistribute_interval != 0:
            return
        self.redistribute(module)

    def redistribute(self, module):
        with torch.no_grad():
            # gather number of Gaussians
            member_n_gaussians = [0 for _ in range(self.world_size)]
            torch.distributed.all_gather_object(member_n_gaussians, module.gaussian_model.get_xyz.shape[0])
            if self.global_rank == 0:
                print(f"[rank={self.global_rank}] member_n_gaussians={member_n_gaussians}")

            if min(member_n_gaussians) * self.config.redistribute_threshold >= max(member_n_gaussians):
                print(f"[rank={self.global_rank}] skip redistribution: under threshold")
                return

            print(f"[rank={self.global_rank}] begin redistribution")
            self.random_redistribute(module)

    def random_redistribute(self, module):
        destination = torch.randint(0, self.world_size, (module.gaussian_model.get_xyz.shape[0],), device=module.device)
        count_by_destination = list(torch.bincount(destination, minlength=self.world_size).chunk(self.world_size))

        print(f"[rank={self.global_rank}] destination_count={[i.item() for i in count_by_destination]}")

        # number of gaussians to receive all-to-all
        number_of_gaussians_to_receive = list(torch.zeros((self.world_size,), dtype=count_by_destination[0].dtype, device=module.device).chunk(self.world_size))
        torch.distributed.nn.functional.all_to_all(number_of_gaussians_to_receive, count_by_destination)

        self.optimizer_all2all(destination, number_of_gaussians_to_receive, module.gaussian_model, module.gaussian_optimizers)

        new_number_of_gaussians = module.gaussian_model.get_xyz.shape[0]
        print(f"[rank={self.global_rank}] redistributed: n_gaussians={new_number_of_gaussians}")

        self.on_density_changed()

    def all2all_gaussian_state(self, local_tensor, destination, number_of_gaussians_to_receive):
        output_tensor_list = []
        input_tensor_list = []

        for i in range(self.world_size):
            output_tensor_list.append(torch.empty(
                [number_of_gaussians_to_receive[i]] + list(local_tensor.shape[1:]),
                dtype=local_tensor.dtype,
                device=local_tensor.device,
            ))
            input_tensor_list.append(local_tensor[destination == i])

        torch.distributed.nn.functional.all_to_all(output_tensor_list, input_tensor_list)

        return torch.concat(output_tensor_list, dim=0).contiguous()

    def optimizer_all2all(self, destination, number_of_gaussians_to_receive, gaussian_model, optimizers):
        def invoke_all2all(local_tensor):
            return self.all2all_gaussian_state(local_tensor, destination=destination, number_of_gaussians_to_receive=number_of_gaussians_to_receive)

        optimizable_tensors = {}
        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = invoke_all2all(stored_state["exp_avg"])
                    stored_state["exp_avg_sq"] = invoke_all2all(stored_state["exp_avg_sq"])

                    # replace with new tensor and state
                    del opt.state[group['params'][0]]
                    group["params"][0] = torch.nn.Parameter(invoke_all2all(group["params"][0]).requires_grad_(True))
                    opt.state[group['params'][0]] = stored_state
                else:
                    group["params"][0] = torch.nn.Parameter(invoke_all2all(group["params"][0]).requires_grad_(True))

                optimizable_tensors[group["name"]] = group["params"][0]

        # update
        gaussian_model.properties = optimizable_tensors

    def get_available_outputs(self) -> Dict:
        return {
            "rgb": RendererOutputInfo("render"),
        }
