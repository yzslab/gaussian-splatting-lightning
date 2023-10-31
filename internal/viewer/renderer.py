import internal.renderers as renderers


class ViewerRenderer:
    def __init__(
            self,
            gaussian_model,
            renderer: renderers.Renderer,
            background_color,
    ):
        super().__init__()

        self.gaussian_model = gaussian_model
        self.renderer = renderer
        self.background_color = background_color

    def get_outputs(self, camera, scaling_modifier: float = 1.):
        return self.renderer(
            camera,
            self.gaussian_model,
            self.background_color,
            scaling_modifier=scaling_modifier,
        )["render"]
