import os
import torch
import viser
from internal.utils.sh_utils import RGB2SH


class DistanceMeasurementPanel:
    def __init__(
            self,
            viewer,
            server: viser.ViserServer,
            tab,
    ):
        self.viewer = viewer
        self.server = server

        self.property_backup = None

        self.scale = .1
        self.distant_scale = 1.
        self.used_point_indices = (-1, -2)

        # initial on button clicked
        enable_measurement_button = self.server.gui.add_button("Enable")

        @enable_measurement_button.on_click
        def _(event):
            enable_measurement_button.disabled = True
            enable_measurement_button.visible = False
            assert self.property_backup is None
            with self.server.atomic():
                gaussian_model = self.viewer.gaussian_model
                self.property_backup = []
                for i in self.used_point_indices:
                    self.property_backup.append((
                        gaussian_model.get_means()[i].clone(),
                        gaussian_model.get_scales()[i].clone(),
                        gaussian_model.get_opacities()[i].clone(),
                        gaussian_model.get_shs()[i].clone(),
                    ))

            with tab:
                self.setup_panel(initial_position=event.client.camera.look_at)

    def setup_panel(self, initial_position):
        with self.server.gui.add_folder("Settings"):
            # point scale
            self.scale_number = self.server.gui.add_number(
                "Scale",
                initial_value=self.scale,
                min=0.,
                step=0.0001,
            )

            @self.scale_number.on_update
            def _(event):
                self.scale = self.scale_number.value
                for i in self.used_point_indices:
                    self.update_point(i)
                self.viewer.rerender_for_all_client()

            # distant scale
            self.distance_scale_number = self.server.gui.add_number(
                "Dist Scale",
                initial_value=self.distant_scale,
                min=0.,
                step=0.0001,
            )

            @self.distance_scale_number.on_update
            def _(_):
                self.distant_scale = self.distance_scale_number.value
                update_distances()

        # raw distance and height values
        with self.server.gui.add_folder("Raw"):
            self.distance_number = self.server.gui.add_number(
                "Dist",
                initial_value=0.,
                min=0.,
                step=0.0001,
            )
            self.heigh_number = self.server.gui.add_number(
                "Height",
                initial_value=0.,
                min=0.,
                step=0.0001,
            )

        # scaled
        with self.server.gui.add_folder("Scaled"):
            self.scaled_distance = self.server.gui.add_number(
                "Dist",
                initial_value=0.,
                min=0.,
                step=0.0001,
            )
            self.scaled_heigh = self.server.gui.add_number(
                "Height",
                initial_value=0.,
                min=0.,
                step=0.0001,
            )

        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "measurements"
        )
        os.makedirs(save_dir, exist_ok=True)

        # save
        with self.server.gui.add_folder("Save"):
            self.name_text = self.server.gui.add_text(
                "Name",
                initial_value="",
            )
            self.save_button = self.server.gui.add_button(
                "Save",
            )

            @self.save_button.on_click
            def _(_):
                self.save_button.disabled = True
                save_file_path = os.path.join(save_dir, "{}.pt".format(self.name_text.value))
                torch.save(self.get_point_xyzs(), save_file_path)
                self.save_button.disabled = False

        def get_file_list():
            file_list = [""]
            for i in os.scandir(save_dir):
                if i.name.endswith(".pt"):
                    file_list.append(i.name)
            return file_list

        # load
        with self.server.gui.add_folder("Load"):
            self.file_list = self.server.gui.add_dropdown(
                "Files",
                options=get_file_list(),
            )
            self.refresh_button = self.server.gui.add_button("Refresh")
            self.load_button = self.server.gui.add_button("Load")

            @self.refresh_button.on_click
            def _(_):
                self.file_list.options = get_file_list()

            @self.load_button.on_click
            def _(_):
                if self.file_list.value == "":
                    return
                loaded = torch.load(os.path.join(save_dir, self.file_list.value))
                for idx, index in enumerate(self.used_point_indices):
                    self.update_point(index, xyz=loaded[idx])
                    self.transform_controls[idx].position = loaded[idx].cpu().numpy()
                self.viewer.rerender_for_all_client()
                update_distances()

        # distance label
        # self.distance_label = self.server.scene.add_label(
        #     name="Dist Label",
        #     text="0",
        #     position=initial_position,
        # )

        def update_distances():
            dist, height = self.get_point_distance()
            self.distance_number.value = dist
            self.heigh_number.value = height
            self.scaled_distance.value = dist * self.distant_scale
            self.scaled_heigh.value = height * self.distant_scale

            # self.distance_label.position = self.get_point_xyzs().mean(dim=0).cpu().numpy()
            # self.distance_label.text = "{}".format(dist)

        # point transform
        self.transform_controls = []
        for idx, i in enumerate(self.used_point_indices):
            self.update_point(i, initial_position)
            self.transform_controls.append(self.setup_point_transform_control(idx, i, initial_position, update_distances))

    def update_point(self, index: int, xyz=None):
        with torch.no_grad():
            device = self.viewer.gaussian_model.means[index].device
            if xyz is not None:
                self.viewer.gaussian_model.means[index] = torch.tensor(xyz, dtype=torch.float, device=device)
            self.viewer.gaussian_model.scales[index] = self.viewer.gaussian_model.scale_inverse_activation(torch.tensor(self.scale, dtype=torch.float, device=device))
            self.viewer.gaussian_model.opacities[index] = 100.
            self.viewer.gaussian_model.gaussians["shs"][index, 0, :] = RGB2SH(torch.tensor([1., 0., 0.], dtype=torch.float, device=device))
            self.viewer.gaussian_model.gaussians["shs"][index, 1:, :] = 0.

    def get_point_xyzs(self):
        with torch.no_grad():
            return self.viewer.gaussian_model.means[torch.tensor(self.used_point_indices, dtype=torch.long, device="cuda")]

    def get_point_distance(self):
        xyzs = self.get_point_xyzs()
        up_direction = torch.from_numpy(self.viewer.up_direction).unsqueeze(0).to(dtype=xyzs.dtype, device=xyzs.device)
        projections = torch.sum(xyzs * up_direction, dim=-1)
        # print(xyzs)
        return torch.linalg.norm(xyzs[0] - xyzs[1]).item(), torch.abs(projections[0] - projections[1]).item()

    def setup_point_transform_control(self, idx, index, initial_position, update_distances):
        transform_control = self.server.scene.add_transform_controls(
            "P{}".format(idx),
            scale=0.5,
            position=initial_position
        )

        @transform_control.on_update
        def _(_):
            print("p{}={}".format(idx, transform_control.position))
            self.update_point(index, transform_control.position)
            update_distances()
            self.viewer.rerender_for_all_client()

        return transform_control
