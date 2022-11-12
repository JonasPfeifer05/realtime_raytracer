use std::sync::Arc;


use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::{sync, VulkanLibrary};
use vulkano::command_buffer::{CopyImageToBufferInfo, PrimaryAutoCommandBuffer};
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo};
use vulkano::pipeline::ComputePipeline;
use vulkano::sync::GpuFuture;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::image::{ImageDimensions, StorageImage};
use vulkano::format::Format;
use vulkano::pipeline::Pipeline;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::image::view::ImageView;
use bytemuck::{Zeroable, Pod};

pub struct VulkanHelper {
    command_buffer: Arc<PrimaryAutoCommandBuffer>,
    buffer: Arc<CpuAccessibleBuffer<[u8]>>,
    size_buffer: Arc<CpuAccessibleBuffer<Size>>,
    compute_pipeline: Arc<ComputePipeline>,
    queue: Arc<Queue>,
    device: Arc<Device>,
}

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct Size {
    x: u32,
    y: u32,
}

impl VulkanHelper {
    pub fn resize(&mut self, width: u32, height: u32) {
        let image = StorageImage::new(
            self.device.clone(),
            ImageDimensions::Dim2d {
                width,
                height,
                array_layers: 1, // images can be arrays of layers
            },
            Format::R8G8B8A8_UNORM,
            Some(self.queue.queue_family_index()),
        ).unwrap();

        let buffer =
            CpuAccessibleBuffer::from_iter(self.device.clone(), BufferUsage {
                transfer_dst: true,
                ..Default::default()
            }, false, (0..width * height * 4).map(|_| 0u8)).expect("failed to create buffer");


        let view = ImageView::new_default(image.clone()).unwrap();


        let layout = self.compute_pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::image_view(0, view.clone())], // 0 is the binding
        ).unwrap();


        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        ).unwrap();

        builder
            .bind_pipeline_compute(self.compute_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0, // 0 is the index of our set
                set,
            )
            .dispatch([((width * 2) as f32 / 8 as f32).ceil() as u32 + 1, ((height * 2) as f32 / 8 as f32).ceil() as u32 + 1, 1])
            .unwrap()
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                image.clone(),
                buffer.clone(),
            ))
            .unwrap();

        let command_buffer = Arc::new(builder.build().unwrap());

        self.command_buffer = command_buffer;
        self.buffer = buffer;
    }

    pub fn render_frame(&self) -> Arc<CpuAccessibleBuffer<[u8]>> {
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), self.command_buffer.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        self.buffer.clone()
    }

    pub fn new(width: u32, height: u32) -> Self {
        let library = VulkanLibrary::new().expect("Didn´t finde local vulkan library/dll!");
        println!("Using vulkan version: {}", library.api_version());


        let instance = Instance::new(library, InstanceCreateInfo::default()).expect("Failed to create instance!");


        let physical = instance
            .enumerate_physical_devices()
            .expect("Could not enumerate devices!")
            .next()
            .expect("No device available!");


        let queue_family_index = physical
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_, q)| q.queue_flags.compute)
            .expect("Couldn´t find find a compute queue family!") as u32;


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


        let queue = queues.next().unwrap();


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
            }, false, (0..width * height * 4).map(|_| 0u8)).expect("failed to create buffer");

        let size_buffer =
            CpuAccessibleBuffer::from_data(device.clone(), BufferUsage {
                uniform_buffer: true,
                ..Default::default()
            }, false, Size {x: width, y: height}).expect("failed to create buffer");

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

            float aspect_ratio = float(imageSize(img).x) / float(imageSize(img).y);
            uint image_width = imageSize(img).x;
            uint image_height = imageSize(img).y;

            // Camera

            float viewport_height = 2.0;
            float viewport_width = aspect_ratio * viewport_height;
            float focal_length = 1.0;

            vec3 origin = vec3(0, 0, 0);
            vec3 horizontal = vec3(viewport_width, 0, 0);
            vec3 vertical = vec3(0, viewport_height, 0);
            vec3 lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);

            const uint sphere_size = 2;
            const Sphere[] spheres = Sphere[sphere_size](
                Sphere(vec3(0.0,-100.0,-1.0),100),
                Sphere(vec3(0.0,0.0,-1.0),0.5)
            );

            vec3 calculate_color(in uvec2 xy) {
                float u = float(xy.x) / (image_width-1.0);
                float v = abs(float(xy.y) / (image_height-1.0)-1);
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
        let shader = cs::load(device.clone()).expect("Failed to create shader!");


        let compute_pipeline = ComputePipeline::new(
            device.clone(),
            shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        ).expect("Failed to create compute pipeline!");


        let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [
                WriteDescriptorSet::image_view(0, view.clone()),
                //WriteDescriptorSet::buffer(1, size_buffer.clone())
            ] // 0 is the binding
        ).unwrap();


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
            .dispatch([((width * 2) as f32 / 8 as f32).ceil() as u32 + 1, ((height * 2) as f32 / 8 as f32).ceil() as u32 + 1, 1])
            .unwrap()
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                image.clone(),
                buffer.clone(),
            ))
            .unwrap();

        let command_buffer = Arc::new(builder.build().unwrap());


        VulkanHelper {
            command_buffer,
            buffer,
            size_buffer,
            compute_pipeline,
            queue,
            device,
        }
    }
}