mod visual;
mod vulkan;



use std::time::{Instant};
use sdl2::event::{Event, WindowEvent};
use crate::visual::display::Display;
use crate::vulkan::VulkanHelper::VulkanHelper;


fn main() {
    start(1080, 920).unwrap();
}

fn start(width: u32, height: u32) -> Result<(), String>{
    let helper = VulkanHelper::new(width, height);
    render_loop(width, height, helper);

    Ok(())
}

fn render_loop(width: u32, height: u32, mut helper: VulkanHelper) {
    let display = Display::new(width, height).unwrap();

    let mut delta_time;
    let mut start;
    let mut counter = 0;

        let mut events = display.get_event_pump();

        'render_loop: loop {
            // SAVE TIME FOR DELTA TIME
            start = Instant::now();

            for event in events.poll_iter() {
                // HANDLE EVENTS
                match event {
                    Event::Window { win_event: WindowEvent::Resized(width, height), .. } => {
                        helper.resize(width as u32, height as u32);
                    }
                    Event::Window { win_event: WindowEvent::Close, .. } => { break 'render_loop; }
                    _ => {}
                }
            }

            // UPDATING SCREEN
            {
                // GETTING THE PIXEL BUFFER OF THE WINDOW
                let bytes_buffer = helper.render_frame();
                let calculated_bytes = bytes_buffer.read().unwrap();

                let mut surface = display.get_surface(&events);

                let bytes = surface.without_lock_mut().unwrap();


                /*for i in 0..bytes.len() {
                    bytes[i] = ((calculated_bytes[i] as u32+bytes[i] as u32)/2) as u8;
                }*/

                bytes.clone_from_slice(&*calculated_bytes);

                // UPDATE WINDOW
                let _ = surface.update_window().unwrap();
            }

        // STOPPING TIME FOR DELTA TIME
        delta_time = start.elapsed().as_micros();

        counter += delta_time;
        if counter > 1000000 {
            counter = 0;
            println!("ms/frame: {}, fps: {}", delta_time as f32 / 1000.0, 1.0 / (delta_time as f64 / 1000000.0));
        }
    }
}

fn write_u8_pixel(pixel: (u8, u8, u8), index: usize, buffer: &mut [u8]) {
    buffer[index * 4] = pixel.2;        // BLUE
    buffer[index * 4 + 1] = pixel.1;    // GREEN
    buffer[index * 4 + 2] = pixel.0;    // RED
    buffer[index * 4 + 3] = 0;          // ALPHA?
}

fn pixel_f32_to_u8(pixel: (f32, f32, f32)) -> (u8, u8, u8) {
    return ((pixel.0 * 255.0) as u8, (pixel.0 * 255.0) as u8, (pixel.0 * 255.0) as u8);
}
