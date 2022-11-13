use std::ops::Deref;
use std::sync::Arc;


use vulkano::instance::{Instance, InstanceCreateInfo};

use vulkano::{sync, sync::GpuFuture, VulkanLibrary};

use vulkano::command_buffer::{CopyImageToBufferInfo, PrimaryAutoCommandBuffer,AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};

use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo};

use vulkano::pipeline::{PipelineBindPoint, ComputePipeline, Pipeline};
use vulkano::descriptor_set::{DescriptorSetCreationError, PersistentDescriptorSet, WriteDescriptorSet};

use vulkano::image::{view::ImageView, ImageDimensions, StorageImage};
use vulkano::format::Format;

use bytemuck::{Zeroable, Pod};

pub struct VulkanHelper {
    command_buffer: Arc<PrimaryAutoCommandBuffer>,
    buffer: Arc<CpuAccessibleBuffer<[u8]>>,
    camera:Arc<CpuAccessibleBuffer<Camera>>,
    compute_pipeline: Arc<ComputePipeline>,
    queue: Arc<Queue>,
    device: Arc<Device>,
}

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
struct PodVec3(f64,f32,f32);

#[repr(C)]
#[derive(Default, Copy, Clone, Zeroable, Pod)]
pub struct Camera {
    frame: u32,
    width: u32,
    height: u32,
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    lower_left_corner_x: f32,
    lower_left_corner_y: f32,
    lower_left_corner_z: f32,
    horizontal_x: f32,
    horizontal_y: f32,
    horizontal_z: f32,
    vertical_x: f32,
    vertical_y: f32,
    vertical_z: f32,
}

impl Camera {
    pub fn new(width: u32, height: u32) -> Self {
        let aspect_ratio = width as f32 / height as f32;
        let viewport_height = 2.0;
        let viewport_width = viewport_height * aspect_ratio;
        let focal_length = 1.0;

        let origin: [f32; 3] = [0.0, 0.0, 0.0];

        let mut lower_left_corner = origin;
        lower_left_corner[0] -= viewport_width / 2.0;
        lower_left_corner[1] -= viewport_height / 2.0;
        lower_left_corner[2] -= focal_length;

        Self {
            frame: 0,
            width,
            height,
            origin_x: origin[0],
            origin_y: origin[1],
            origin_z: origin[2],
            lower_left_corner_x: lower_left_corner[0],
            lower_left_corner_y: lower_left_corner[1],
            lower_left_corner_z: lower_left_corner[2],
            horizontal_x: viewport_width,
            horizontal_y: 0.0,
            horizontal_z: 0.0,
            vertical_x: 0.0,
            vertical_y: viewport_height,
            vertical_z: 0.0,
        }
    }
}

impl VulkanHelper {
    pub fn next_frame(&mut self) {
        self.camera.write().unwrap().frame += 1;
    }

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

        let view = ImageView::new_default(image.clone()).unwrap();

        let buffer =
            CpuAccessibleBuffer::from_iter(self.device.clone(), BufferUsage {
                transfer_dst: true,
                ..Default::default()
            }, false, (0..width * height * 4).map(|_| 0u8)).expect("failed to create buffer");

        let camera =
            CpuAccessibleBuffer::from_data(self.device.clone(), BufferUsage {
                uniform_buffer: true,
                ..Default::default()
            }, false, Camera::new(width,height)).expect("failed to create buffer");


        let layout = self.compute_pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [
                WriteDescriptorSet::image_view(0, view.clone()),
                WriteDescriptorSet::buffer(1, camera.clone()),
            ] // 0 is the binding
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

        let camera =
            CpuAccessibleBuffer::from_data(device.clone(), BufferUsage {
                uniform_buffer: true,
                ..Default::default()
            }, false, Camera::new(width, height)).expect("failed to create buffer");

        let view = ImageView::new_default(image.clone()).unwrap();

        mod cs {
        vulkano_shaders::shader! {
        ty: "compute",
        src: "
            #version 450

            float PHI = 1.61803398874989484820459;  // Φ = Golden Ratio

            float gold_noise(in vec2 xy, in float seed){
                   return fract(tan(distance(xy*PHI, xy)*seed)*xy.x);
            }

            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

            layout(set = 0, binding = 0, rgba8) uniform image2D img;


            struct Ray {
                vec3 origin;
                vec3 direction;
            };
            vec3 ray_at(in Ray ray, in float t) {
                return ray.origin + ray.direction * t;
            }

            struct CameraReal {
                vec3 origin;
                vec3 lower_left_corner;
                vec3 horizontal;
                vec3 vertical;
            };

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


            // Camera

            struct Camera {
                uint frame;
                uint width;
                uint height;
                float origin_x;
                float origin_y;
                float origin_z;
                float lower_left_corner_x;
                float lower_left_corner_y;
                float lower_left_corner_z;
                float horizontal_x;
                float horizontal_y;
                float horizontal_z;
                float vertical_x;
                float vertical_y;
                float vertical_z;
            };

            layout(set = 0, binding = 1) uniform CameraData {
                Camera cam;
            } camera_data;

            Ray get_ray(in CameraReal real, in float u, in float v) {
                return Ray(real.origin, real.lower_left_corner + u*real.horizontal + v*real.vertical - real.origin);
            }

            const uint sphere_size = 2;
            const Sphere[] spheres = Sphere[sphere_size](
                Sphere(vec3(0.0,-100.0,-1.0),100),
                Sphere(vec3(0.0,0.0,-1.0),0.5)
            );

            vec3 calculate_color(in uvec2 xy, float microstep) {
                float u = float(xy.x) / (camera_data.cam.width-1.0);
                u += fract(gold_noise(xy.xy, camera_data.cam.frame+microstep)) / float(camera_data.cam.width);
                float v = abs(float(xy.y) / (camera_data.cam.height-1.0)-1);
                v += fract(gold_noise(xy.xy, camera_data.cam.frame+microstep+0.5)) / float(camera_data.cam.height);

                CameraReal real = CameraReal(
                    vec3(camera_data.cam.origin_x,camera_data.cam.origin_y,camera_data.cam.origin_z),
                    vec3(camera_data.cam.lower_left_corner_x,camera_data.cam.lower_left_corner_y,camera_data.cam.lower_left_corner_z),
                    vec3(camera_data.cam.horizontal_x,camera_data.cam.horizontal_y,camera_data.cam.horizontal_z),
                    vec3(camera_data.cam.vertical_x,camera_data.cam.vertical_y,camera_data.cam.vertical_z));

                Ray ray = get_ray(real, u, v);

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
                vec3 color;
                color = calculate_color(gl_GlobalInvocationID.xy, 0.0);

                imageStore(img, ivec2(gl_GlobalInvocationID.xy), (imageLoad(img, ivec2(gl_GlobalInvocationID.xy)) + vec4(color.zyx, 1.0))/2.0);
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
        let mut set = PersistentDescriptorSet::new(
            layout.clone(),
            [
                WriteDescriptorSet::image_view(0, view.clone()),
                WriteDescriptorSet::buffer(1, camera.clone()),
            ], // 0 is the binding
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
            camera,
            compute_pipeline,
            queue,
            device,
        }
    }
}