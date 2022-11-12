use sdl2::pixels::Color;
use sdl2::render::WindowCanvas;
use sdl2::{EventPump, Sdl};
use sdl2::video::WindowSurfaceRef;

pub struct Display {
    context: Sdl,
    canvas: WindowCanvas,
}

impl Display {
    pub fn new(width: u32, height: u32) -> Result<Self, String> {
        let context = sdl2::init()?;
        let video_subsystem = context.video()?;

        let window = video_subsystem
            .window("Realtime Raytracer", width, height)
            .position_centered()
            .resizable()
            .build()
            .map_err(|e| e.to_string())?;

        let mut canvas = window.into_canvas().build().map_err(|e| e.to_string())?;

        canvas.set_draw_color(Color::WHITE);
        canvas.clear();
        canvas.present();

        Ok(Self { context, canvas })
    }

    pub fn get_event_pump(&self) -> EventPump {
        self.context.event_pump().unwrap()
    }

    pub fn get_size(&self) -> (u32,u32) {
        self.canvas.output_size().unwrap()
    }

    pub fn get_surface<'a>(&'a self, event_pump: &'a EventPump) -> WindowSurfaceRef {
        self.canvas.window().surface(event_pump).unwrap()
    }
}