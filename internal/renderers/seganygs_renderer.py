import os
import traceback
from typing import Dict

import re
import torch
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import spherical_harmonics
from .renderer import Renderer
from .gsplat_renderer import GSPlatRenderer, DEFAULT_ANTI_ALIASED_STATUS, DEFAULT_BLOCK_SIZE
from .gsplat_contrastive_feature_renderer import GSplatContrastiveFeatureRenderer
from ..cameras import Camera
from ..models.gaussian_model import GaussianModel


class SegAnyGSRenderer(Renderer):
    SEGMENT_RENDER_MODE_COLORED = 0
    SEGMENT_RENDER_MODE_SEGMENT_OUT = 1

    def __init__(
            self,
            semantic_features: torch.Tensor,
            scale_gate: torch.nn.Module,
            anti_aliased: bool = DEFAULT_ANTI_ALIASED_STATUS,
    ):
        super().__init__()

        self.anti_aliased = anti_aliased

        # move to cuda first
        self.semantic_features = semantic_features.cuda()
        self.scale_gate = scale_gate.cuda()
        self.pca_projection_matrix = self._get_pca_projection_matrix(self.semantic_features)
        self.pca_colors = ((self.semantic_features @ self.pca_projection_matrix) * 0.5 + 0.5).clamp(0., 1.)

        self.segment_mask = None
        self.feature_list = []
        self.segment_render_mode = self.SEGMENT_RENDER_MODE_SEGMENT_OUT

        # calculate some scale relative values
        self._update_scale(1.)

        self.cluster_result_save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "clusters",
        )
        self.cluster_result = None
        self.cluster_color = None

        # reduce CUDA memory consumption
        self.semantic_features = self.semantic_features.cpu()
        torch.cuda.empty_cache()

        self.color_producers = {
            "rgb": self._shs_to_rgb,
            "depth": self._depth_as_color,
            "pca3d": self._pca_as_color,
            "scale_gated_pca3d": self._scale_gated_pca_as_color,
            "cluster3d": self._cluster_as_color,
            "segment3d": self._segment_as_color,
            "segment3d_out": self._segment_out,
        }

        self.available_output_types = {
            "rgb": "rgb",
            "depth": "depth",
            "pca3d": "pca3d",
            "scale_gated_pca3d": "pca3d_scale_gated",
            "cluster3d": "cluster3d",
            "segment3d": "segment3d",
            "segment3d_out": "segment3d_out",
        }

    def _get_pca_projection_matrix(self, semantic_features, n_components: int = 3):
        randint = torch.randint(0, semantic_features.shape[0], [200_000])
        X = semantic_features[randint, :]

        n = X.shape[0]
        mean = torch.mean(X, dim=0)
        X = X - mean
        covariance_matrix = (1 / n) * torch.matmul(X.T, X).float()  # An old torch bug: matmul float32->float16,
        eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)
        idx = torch.argsort(-eigenvalues.real)
        eigenvectors = eigenvectors.real[:, idx]
        proj_mat = eigenvectors[:, 0:n_components]

        return proj_mat

    def _update_scale(self, scale):
        self.scale = scale

        semantic_features = self.semantic_features.cuda()

        scale_conditioned_semantic_features = torch.nn.functional.normalize(
            semantic_features * self._scale_gate_forward(self.scale),
            dim=-1,
        )

        self.scale_gated_pca_colors = (scale_conditioned_semantic_features @ self.pca_projection_matrix)
        self.scale_gated_pca_colors = self.scale_gated_pca_colors - self.scale_gated_pca_colors.min(dim=0).values
        self.scale_gated_pca_colors = self.scale_gated_pca_colors / self.scale_gated_pca_colors.max(dim=0).values

        self.scale_conditioned_semantic_features = scale_conditioned_semantic_features.cpu()

        self._segment()

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            render_types: list = None,
            **kwargs,
    ):
        project_results = GSPlatRenderer.project(
            means3D=pc.get_xyz,
            scales=pc.get_scaling,
            rotations=pc.get_rotation,
            viewpoint_camera=viewpoint_camera,
            scaling_modifier=scaling_modifier,
        )

        opacities = pc.get_opacity
        if self.anti_aliased is True:
            comp = project_results[4]
            opacities = opacities * comp[:, None]

        img_height = int(viewpoint_camera.height.item())
        img_width = int(viewpoint_camera.width.item())

        outputs = {}

        for i in render_types:
            colors, rasterize_bg_color, new_opacities = self.color_producers[i](project_results, pc, viewpoint_camera, bg_color, opacities)
            outputs[self.available_output_types[i]] = self.rasterize(project_results, img_height=img_height, img_width=img_width, colors=colors, bg_color=rasterize_bg_color, opacities=new_opacities)

        return outputs

    def rasterize(self, project_results, img_height, img_width, colors, bg_color, opacities):
        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_results

        return rasterize_gaussians(  # type: ignore
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,  # type: ignore
            colors,
            opacities,
            img_height=img_height,
            img_width=img_width,
            block_width=DEFAULT_BLOCK_SIZE,
            background=bg_color,
            return_alpha=False,
        ).permute(2, 0, 1)  # type: ignore

    def _shs_to_rgb(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        means3D = pc.get_xyz
        viewdirs = means3D.detach() - viewpoint_camera.camera_center  # (N, 3)
        # viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
        rgbs = spherical_harmonics(pc.active_sh_degree, viewdirs, pc.get_features)
        rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore

        return rgbs, bg_color, opacities

    def _depth_as_color(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        return project_results[1].unsqueeze(-1), torch.zeros((1,), dtype=torch.float, device=bg_color.device), opacities

    def _pca_as_color(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        return self.pca_colors, bg_color, opacities

    def _scale_gated_pca_as_color(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        return self.scale_gated_pca_colors, bg_color, opacities

    def _cluster_as_color(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        if self.cluster_color is None:
            self._cluster_in_3d()

        return self.cluster_color, bg_color, opacities

    def _segment_as_color(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        colors, bg_color, opacities = self._shs_to_rgb(project_results, pc, viewpoint_camera, bg_color, opacities)
        if self.segment_mask is not None:
            colors[self.segment_mask] = torch.tensor([0., 1., 1.], dtype=torch.float, device=bg_color.device)
        return colors, bg_color, opacities

    def _segment_out(self, project_results, pc: GaussianModel, viewpoint_camera, bg_color, opacities):
        colors, bg_color, opacities = self._shs_to_rgb(project_results, pc, viewpoint_camera, bg_color, opacities)
        if self.segment_mask is not None:
            opacities = opacities * self.segment_mask.unsqueeze(-1)
        return colors, bg_color, opacities

    def _cluster_in_3d(self):
        print("clustering...")
        # get scale gated features
        scale_conditioned_point_features = self.scale_conditioned_semantic_features
        # select points randomly
        normed_sampled_point_features = scale_conditioned_point_features[torch.rand(scale_conditioned_point_features.shape[0]) > 0.98]

        import hdbscan
        import numpy as np

        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.01)
        cluster_labels = clusterer.fit_predict(normed_sampled_point_features.detach().cpu().numpy())
        print(np.unique(cluster_labels))

        cluster_centers = torch.zeros(len(np.unique(cluster_labels)) - 1, normed_sampled_point_features.shape[-1])
        for i in range(1, len(np.unique(cluster_labels))):
            cluster_centers[i - 1] = torch.nn.functional.normalize(normed_sampled_point_features[cluster_labels == i - 1].mean(dim=0), dim=-1)

        seg_score = torch.einsum('nc,bc->bn', cluster_centers.cpu(), scale_conditioned_point_features.cpu())

        label_to_color = np.random.rand(1000, 3)
        point_colors = label_to_color[seg_score.argmax(dim=-1).cpu().numpy()]
        point_colors[seg_score.max(dim=-1)[0].detach().cpu().numpy() < 0.5] = (0, 0, 0)

        self.cluster_color = torch.tensor(point_colors, dtype=torch.float, device="cuda")

        self.cluster_result = {
            "cluster_labels": cluster_labels,
            "cluster_centers": cluster_centers,
            "seg_score": seg_score,
            "label_to_color": label_to_color,
            "cluster_color": self.cluster_color,
        }

        print("cluster finished")

    def _show_message(self, client, message: str):
        with client.add_gui_modal("Message") as modal:
            client.add_gui_markdown(message)
            close_button = client.add_gui_button("Close")

            @close_button.on_click
            def _(_) -> None:
                try:
                    modal.close()
                except:
                    pass

    def _scale_gate_forward(self, scale):
        return self.scale_gate(
            torch.tensor([scale], dtype=torch.float, device="cuda")
        )

    def _query_feature_preprocess(self, query_feature):
        scale_gated_query_feature = query_feature * self._scale_gate_forward(self.scale).to(device=query_feature.device)
        scale_gated_query_feature = torch.nn.functional.normalize(scale_gated_query_feature, dim=-1)
        return scale_gated_query_feature

    def _segment(self):
        if len(self.feature_list) == 0:
            return

        scale_conditioned_semantic_features = self.scale_conditioned_semantic_features.cuda()

        query_features = torch.stack(self.feature_list)
        query_features = self._query_feature_preprocess(query_features).T
        similarities = (torch.einsum("NC,CA->NA", scale_conditioned_semantic_features, query_features) + 1.) / 2.  # [N_points, N_query_features]
        similarities = torch.pow(similarities + 1e-9, self.similarity_score_gamma.value)
        mask = (similarities >= self.similarity_score_number.value).sum(dim=-1) > 0
        self.segment_mask = mask

    def _add_segment(self, query_feature):
        if self.segment_mask is None:
            self.segment_mask = torch.zeros((self.scale_conditioned_semantic_features.shape[0],), dtype=torch.bool, device="cuda")

        query_feature = self._query_feature_preprocess(query_feature)
        scale_conditioned_semantic_features = self.scale_conditioned_semantic_features.cuda()
        similarities = (torch.einsum('C,NC->N', query_feature, scale_conditioned_semantic_features) + 1.) / 2.
        similarities = torch.pow(similarities + 1e-9, self.similarity_score_gamma.value)
        self.segment_mask = torch.logical_or(self.segment_mask, similarities >= self.similarity_score_number.value)

    def _setup_segment(self, viewer, server):
        from internal.viewer.client import ClientThread

        feature_map_render = GSplatContrastiveFeatureRenderer()
        self._feature_map = None
        self.feature_list = []

        point_number = server.add_gui_number(
            label="Prompt",
            initial_value=0,
            disabled=True,
        )
        selected_point_number = server.add_gui_number(
            label="Maksed",
            initial_value=0,
            disabled=True,
        )
        similarity_score_number = server.add_gui_number(
            label="Similarity Score",
            initial_value=0.8,
            min=-1.,
            max=1.,
            step=0.001,
        )
        self.similarity_score_number = similarity_score_number
        similarity_score_gamma = server.add_gui_number(
            label="Score Gamma",
            initial_value=1.,
            min=0.,
            max=10.,
            step=0.01,
            hint="Smaller the gamma, more the high score"
        )
        self.similarity_score_gamma = similarity_score_gamma

        @similarity_score_number.on_update
        @similarity_score_gamma.on_update
        def _(_):
            with server.atomic():
                self._segment()
            viewer.rerender_for_all_client()

        enable_click_mode_button = server.add_gui_button("Enter Click Mode")
        disable_click_mode_button = server.add_gui_button("Exit Click Mode", visible=False, color="red")

        @enable_click_mode_button.on_click
        def _(event):
            enable_click_mode_button.visible = False
            disable_click_mode_button.visible = True

            self._switch_renderer_output_type(viewer, "segment3d")

            max_res = viewer.max_res_when_static.value
            camera = ClientThread.get_camera(
                event.client.camera,
                image_size=max_res,
            ).to_device(viewer.device)

            self._feature_map = feature_map_render(
                viewpoint_camera=camera,
                pc=viewer.viewer_renderer.gaussian_model,
                bg_color=torch.zeros((self.semantic_features.shape[-1],), dtype=torch.float, device=viewer.device),
                semantic_features=self.semantic_features.to(device=viewer.device),
            )["render"].permute(1, 2, 0)

            @server.on_scene_pointer(event_type="click")
            def on_scene_click(event):
                x, y = round(event.screen_pos[0][0] * (self._feature_map.shape[1] - 1)), round(event.screen_pos[0][1] * (self._feature_map.shape[0] - 1))
                print(f"x={x}, y={y}")

                feature = self._feature_map[y, x]
                feature = feature / (torch.norm(feature) + 1e-9)
                self.feature_list.append(feature)
                self._add_segment(feature)
                selected_point_number.value = self.segment_mask.sum().item()
                point_number.value += 1
                viewer.rerender_for_all_client()

        @disable_click_mode_button.on_click
        def _(event):
            server.remove_scene_pointer_callback()
            self._feature_map = None
            enable_click_mode_button.visible = True
            disable_click_mode_button.visible = False

        # clear points
        clear_prompt_point_button = server.add_gui_button("Clear Prompt Points", color="red")

        @clear_prompt_point_button.on_click
        def _(_):
            with server.atomic():
                self.feature_list.clear()
                self.segment_mask = None
                point_number.value = 0
                selected_point_number.value = 0
            viewer.rerender_for_all_client()

    def _setup_cluster(self, viewer, server):
        clustering_button = server.add_gui_button(
            label="Clustering...",
            disabled=True,
            visible=False,
        )
        cluster_button = server.add_gui_button(
            label="Re-Cluster in 3D",
        )

        @cluster_button.on_click
        def _(_):
            cluster_button.visible = False
            clustering_button.visible = True
            with server.atomic():
                self._cluster_in_3d()
            cluster_button.visible = True
            clustering_button.visible = False

            # switch output type to cluster3d
            self._switch_renderer_output_type(viewer, "cluster3d")

    def _switch_renderer_output_type(self, viewer, type):
        viewer.viewer_renderer.output_type_dropdown.value = type
        viewer.viewer_renderer._set_output_type(type, self.available_output_types[type])
        viewer.rerender_for_all_client()

    def _setup_save_cluster(self, viewer, server):
        save_name_text = server.add_gui_text(label="Name", initial_value="")
        save_cluster_button = server.add_gui_button(label="Save")

        @save_cluster_button.on_click
        def _(event):
            save_cluster_button.disabled = True
            with server.atomic():
                try:
                    output_path = self.save_cluster_results(save_name_text.value)
                    message_text = f"Saved to '{output_path}'"
                except Exception as e:
                    message_text = str(e)
                    traceback.print_exc()
            save_cluster_button.disabled = False
            self._show_message(event.client, message_text)

    def _setup_load_cluster(self, viewer, server):
        reload_file_list_button = server.add_gui_button(
            label="Refresh",
        )
        cluster_result_file_dropdown = server.add_gui_dropdown(
            label="File",
            options=self._scan_cluster_files(),
            initial_value="",
        )
        load_cluster_button = server.add_gui_button(
            label="Load",
        )

        @reload_file_list_button.on_click
        def _(_):
            cluster_result_file_dropdown.options = self._scan_cluster_files()

        @load_cluster_button.on_click
        def _(event):
            match = re.search("^[a-zA-Z0-9_\-.]+\.pt$", cluster_result_file_dropdown.value)
            if not match:
                self._show_message(event.client, "Invalid filename")
                return

            loaded = False

            load_cluster_button.disabled = True
            with server.atomic():
                cluster_result = torch.load(os.path.join(self.cluster_result_save_dir, cluster_result_file_dropdown.value))
                if cluster_result["cluster_color"].shape[0] == self.semantic_features.shape[0]:
                    cluster_result["cluster_color"] = cluster_result["cluster_color"].cuda()
                    self.cluster_result = cluster_result
                    self.cluster_color = cluster_result["cluster_color"]
                    loaded = True
                else:
                    self._show_message(event.client, "File not match to current scene")

            # switch output type to cluster3d
            if loaded is True:
                viewer.viewer_renderer.output_type_dropdown.value = "cluster3d"
                viewer.viewer_renderer._set_output_type("cluster3d", self.available_output_types["cluster3d"])
                viewer.rerender_for_all_client()
            load_cluster_button.disabled = False

    def setup_tabs(self, viewer, server, tabs):
        with tabs.add_tab("Semantic"):
            scale_slider = server.add_gui_number(
                "Scale",
                min=0.,
                max=1.,
                step=0.001,
                initial_value=self.scale,
            )

            @scale_slider.on_update
            def _(_):
                with server.atomic():
                    self._update_scale(scale_slider.value)
                    viewer.rerender_for_all_client()

            with server.add_gui_folder("Segment"):
                self._setup_segment(viewer, server)

            with server.add_gui_folder("Cluster"):
                self._setup_cluster(viewer, server)
                server.add_gui_markdown("")
                with server.add_gui_folder("Save Cluster"):
                    self._setup_save_cluster(viewer, server)
                with server.add_gui_folder("Load Cluster"):
                    self._setup_load_cluster(viewer, server)

    def _scan_cluster_files(self):
        file_list = []
        try:
            for i in os.listdir(self.cluster_result_save_dir):
                if i.endswith(".pt"):
                    file_list.append(i)
        except:
            pass
        return file_list

    def save_cluster_results(self, name):
        if self.cluster_result is None:
            raise RuntimeError("Please click 'Re-Cluster in 3D' first")

        match = re.search("^[a-zA-Z0-9_\-.]+$", name)
        if match:
            output_path = os.path.join(self.cluster_result_save_dir, f"{name}.pt")
            if os.path.exists(output_path):
                raise RuntimeError("File already exists")

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(self.cluster_result, output_path)

            return output_path
        else:
            raise RuntimeError("Invalid name")

    def get_available_output_types(self) -> Dict:
        return self.available_output_types

    def is_type_depth_map(self, t: str) -> bool:
        return t == "depth"
