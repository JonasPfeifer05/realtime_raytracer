mod visual;


use std::sync::Arc;
use std::time::{Instant};
use sdl2::event::{Event, WindowEvent};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::{sync, VulkanLibrary};
use vulkano::command_buffer::{CopyImageToBufferInfo, PrimaryAutoCommandBuffer};
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo};
use vulkano::pipeline::ComputePipeline;
use vulkano::sync::GpuFuture;
use crate::visual::display::Display;
use vulkano::buffer::{BufferAccess, BufferContents, BufferUsage, CpuAccessibleBuffer};
use vulkano::image::{ImageDimensions, StorageImage};
use vulkano::format::Format;
use vulkano::pipeline::Pipeline;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::image::view::ImageView;


fn main() {
    start(1080, 920).unwrap();
}

fn start(width: u32, height: u32) -> Result<(), String>{
    let gpu = prepareVulkan(width, height);

    /*
    let buffer = gpu.calculatePixels();
    let bytes_buffer= buffer.read().unwrap();
    for i in 0..10 {
        println!("{:?}", bytes_buffer.get(i));
    }
     */
    startRendering(width, height, gpu);

    Ok(())
}

struct VulkanGPU {
    command_buffer: Arc<PrimaryAutoCommandBuffer>,
    buffer: Arc<CpuAccessibleBuffer<[u8]>>,
    image: Arc<StorageImage>,
    queue: Arc<Queue>,
    device: Arc<Device>,
}

impl VulkanGPU {
    pub fn calculatePixels(&self) -> Arc<CpuAccessibleBuffer<[u8]>> {
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), self.command_buffer.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        self.buffer.clone()
    }
}

fn prepareVulkan(width: u32, height: u32) -> VulkanGPU {
    // As with other examples, the first step is to create an instance.
    println!("Getting library");
    let library = VulkanLibrary::new().expect("Didn´t finde local vulkan library/dll!");
    println!("Found vulkan version: {}", library.api_version());


    println!("Getting instance");
    let instance = Instance::new(library, InstanceCreateInfo::default()).expect("Failed to create instance!");


    println!("Getting physical");
    let physical = instance
        .enumerate_physical_devices()
        .expect("Could not enumerate devices!")
        .next()
        .expect("No device available!");


    for family in physical.queue_family_properties() {
        println!("Found a queue family with {:?} queue(s)", family.queue_count);
    }
    let queue_family_index = physical
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_, q)| q.queue_flags.compute)
        .expect("Couldn´t find find a compute queue family!") as u32;


    println!("Getting device");
    let (device, mut queues) = Device::new(
        physical,
        DeviceCreateInfo {
            // here we pass the desired queue family to use by index
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
        .expect("failed to create device");


    println!("Getting queue");
    let queue = queues.next().unwrap();


    println!("Creating buffer");
    let image = StorageImage::new(
        device.clone(),
        ImageDimensions::Dim2d {
            width,
            height,
            array_layers: 1, // images can be arrays of layers
        },
        Format::R8G8B8A8_UNORM,
        Some(queue.queue_family_index()),
    ).unwrap();

    let buffer =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage {
            transfer_dst: true,
            ..Default::default()
        }, false, (0..1080*920*4).map(|_| 0u8)).expect("failed to create buffer");


    let view = ImageView::new_default(image.clone()).unwrap();


    mod cs {
        vulkano_shaders::shader! {
        ty: "compute",
        src: "
            #version 450

            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

            layout(set = 0, binding = 0, rgba8) uniform image2D img;

            struct Ray {
                vec3 origin;
                vec3 direction;
            };
            vec3 ray_at(in Ray ray, in float t) {
                return ray.origin + ray.direction * t;
            }

            struct HitRecord {
                vec3 point;
                vec3 normal;
                float t;
                bool front_face;
            };
            void set_face_normal(out HitRecord record, in Ray ray, in vec3 outward_normal) {
                record.front_face = dot(ray.direction, outward_normal) < 0;
                if (record.front_face) {
                    record.normal = outward_normal;
                } else {
                    record.normal = -1.0 * outward_normal;
                }
            }

            struct Sphere {
                vec3 center;
                float radius;
            };

            bool hit_sphere(in Sphere sphere, in Ray ray, in float t_min, in float t_max, out HitRecord record) {
                vec3 oc = ray.origin - sphere.center;
                float a = ray.direction.x*ray.direction.x+ray.direction.y*ray.direction.y+ray.direction.z*ray.direction.z;
                float half_b = dot(oc, ray.direction);
                float c = (oc.x*oc.x+oc.y*oc.y+oc.z*oc.z) - sphere.radius*sphere.radius;
                float discriminant = half_b*half_b - a*c;

                if (discriminant < 0) {
                    return false;
                }
                float sqrtd = sqrt(discriminant);

                float root = (-half_b - sqrtd) / a;
                if (root < t_min || t_max < root) {
                    root = (-half_b + sqrtd) / a;
                    if (root < t_min || t_max < root)
                        return false;
                }

                record.t = root;
                record.point = ray_at(ray, record.t);
                vec3 outward_normal = (record.point - sphere.center) / sphere.radius;
                set_face_normal(record, ray, outward_normal);

                return true;
            }

            const float aspect_ratio = 1080.0 / 920.0;
            const uint image_width = 1080;
            const uint image_height = uint(float(image_width) / aspect_ratio);

            // Camera

            const float viewport_height = 2.0;
            const float viewport_width = aspect_ratio * viewport_height;
            const float focal_length = 1.0;

            const vec3 origin = vec3(0, 0, 0);
            const vec3 horizontal = vec3(viewport_width, 0, 0);
            const vec3 vertical = vec3(0, viewport_height, 0);
            const vec3 lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);

            const uint sphere_size = 2;
            const Sphere[] spheres = Sphere[sphere_size](
                Sphere(vec3(0.0,-100.0,-1.0),100),
                Sphere(vec3(0.0,0.0,-1.0),0.5)
            );

            vec3 calculate_color(in uvec2 xy) {
                float u = float(xy.x) / (float(imageSize(img).x-1.0));
                float v = abs(float(xy.y) / (float(imageSize(img).y-1.0))-1);
                Ray ray = Ray(origin, lower_left_corner + u*horizontal + v*vertical - origin);

                HitRecord temp_record = HitRecord(vec3(0.0,0.0,0.0),vec3(0.0,0.0,0.0),0.0,false);
                HitRecord record = HitRecord(vec3(0.0,0.0,0.0),vec3(0.0,0.0,0.0),0.0,false);
                bool hit_anything = false;
                float closest = 1.0 / 0.0;

                for (int i = 0; i<sphere_size; i++) {
                    if(hit_sphere(spheres[i], ray, 0, closest, temp_record)) {
                        hit_anything = true;
                        closest = temp_record.t;
                        record = temp_record;
                    }
                }

                if (hit_anything) {
                    return 0.5*(record.normal+vec3(1.0,1.0,1.0));
                }

                vec3 unit = normalize(ray.direction);
                float t = 0.5 * (unit.y + 1.0);

                return (1.0-t)*vec3(1.0,1.0,1.0) + t*vec3(0.5,0.7,1.0);
            }

            void main() {
                vec3 color = calculate_color(gl_GlobalInvocationID.xy);
                imageStore(img, ivec2(gl_GlobalInvocationID.xy), vec4(color.zyx, 1.0));
            }
            "
        }
    }
    println!("Creating shader");
    let shader = cs::load(device.clone()).expect("Failed to create shader!");


    println!("Creating compute pipeline");
    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    ).expect("Failed to create compute pipeline!");


    println!("Binding image view");
    let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        layout.clone(),
        [WriteDescriptorSet::image_view(0, view.clone())], // 0 is the binding
    ).unwrap();


    println!("Creating command buffer");
    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::MultipleSubmit,
    ).unwrap();

    builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0, // 0 is the index of our set
            set,
        )
        .dispatch([((1080*2) as f32/8 as f32).ceil() as u32+1, ((920*2) as f32/8 as f32).ceil() as u32+1, 1])
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image.clone(),
            buffer.clone(),
        ))
        .unwrap();

    let command_buffer = Arc::new(builder.build().unwrap());


    VulkanGPU {
        command_buffer,
        buffer: buffer,
        image,
        queue,
        device
    }
}

fn startRendering(width: u32, height: u32, gpu: VulkanGPU) {
    let display = Display::new(width, height).unwrap();

    let mut delta_time = 0;
    let mut start;
    let mut counter = 0;

        let mut events = display.get_event_pump();

        'render_loop: loop {
            // SAVE TIME FOR DELTA TIME
            start = Instant::now();

            for event in events.poll_iter() {
                // HANDLE EVENTS
                match event {
                    Event::Window { win_event: WindowEvent::Resized(width, height), .. } => {}
                    Event::Window { win_event: WindowEvent::Close, .. } => { break 'render_loop; }
                    _ => {}
                }
            }

            // UPDATING SCREEN
            {
                // GETTING THE PIXEL BUFFER OF THE WINDOW
                let bytes_buffer= gpu.calculatePixels();
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
