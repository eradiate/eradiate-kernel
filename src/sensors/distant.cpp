#include <mitsuba/core/bbox.h>
#include <mitsuba/core/bsphere.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sensor.h>

NAMESPACE_BEGIN(mitsuba)

enum class RayTargetType {Distant, Rectangle, Disk};
enum class RayOriginType {Shape, Rectangle, Disk};

// Forward declaration of specialized DistantSensor
template <typename Float, typename Spectrum, RayTargetType TargetType>
class DistantSensorImpl;

/**!

.. _sensor-distant:

Distant directional sensor (:monosp:`distant`)
----------------------------------------------

.. pluginparameters::

 * - to_world
   - |transform|
   - Sensor-to-world transformation matrix.
 * - direction
   - |vector|
   - Alternative (and exclusive) to `to_world`. Direction from which the
     sensor will be recording in world coordinates.
 * - ray_target_type
   - |string|
   - *Optional.* Specify the ray target sampling strategy. 
     If set to ``"rectangle"``, ray targets will be sampled uniformly from 
     a xy-plane-aligned rectangular surface defined by the parameters 
     ``ray_target_a`` and `` ray_target_b``. If set to ``"disk"``, ray targets will
     be sampled uniformly from a xy-plane-aligned disk defined by the parameters 
     ``ray_target_center`` and ``ray_target_radius``. If set to ``"distant"``, ray targets 
     will be sampled to uniformly cover the entire scene and will be positioned 
     on a bounding sphere. Default: ``"distant"``.
 * - ray_target_a
   - |point|
   - Coordinates in the xy-plane for the rectangular ray target area.
 * - ray_target_b
   - |point|
   - Maximum coordinates in the xy-plane for the rectangular ray target area.
 * - ray_target_center
   - |point|
   - Center point in the xy-plane for the circular ray target area.
 * - ray_target_radius
   - |float|
   - Radius for the circular ray target area
    

This sensor plugin implements a distant directional sensor which records
radiation leaving the scene in a given direction. By default, it records the 
(spectral) radiant flux per unit solid angle leaving the scene in the specified
direction (in unit power per unit solid angle). Rays cast by the sensor are 
distributed uniformly on the cross section of the scene's bounding sphere.

.. warning::

    If this sensor is used with an environment map emitter, it will also record 
    radiant flux coming from the part of emitter appearing through the scene's 
    bounding sphere cross section. Care should be taken notably when using the
    `constant` or `envmap` emitters.

With the ``ray_target_*`` parameters users can define an area parallel to the xy-plane
that the sensor's rays will target. The area can be defined in two ways:

- Setting `ray_target_type` to 'rectangle' lets users define a rectangular area through 
  two points `ray_target_a` and `ray_target_b`. The resulting area will cover the area 
  from the smaller to the larger value in both dimensions.
- Setting `ray_target_type` to 'circle'  lets users define a circular ray_target area 
  through a center `ray_target_center` and a radius `ray_target_radius`.
- If `ray_target_type` is not set, the sensor will target the entire cross section
  of the scene's bounding sphere.

*/

template <typename Float, typename Spectrum>
class DistantSensor final : public Sensor<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Sensor, m_world_transform, m_film)
    MTS_IMPORT_TYPES()

    DistantSensor(const Properties &props) : Base(props), m_props(props) {

        std::string ray_target_type = props.string("ray_target_type", "distant");

        // ray target type to set the templated variant in
        // DistantSensorImpl
        if (ray_target_type == "rectangle") {
            m_ray_target_type = RayTargetType::Rectangle;
        } else if (ray_target_type == "disk") {
            m_ray_target_type = RayTargetType::Disk;
        } else if (ray_target_type == "distant") {
            m_ray_target_type = RayTargetType::Distant;
        } else {
            Throw("Unsupported ray target type!");
        }

        props.mark_queried("ray_target_center");
        props.mark_queried("ray_target_radius");
        props.mark_queried("ray_target_a");
        props.mark_queried("ray_target_b");
        props.mark_queried("direction");
        props.mark_queried("to_world");

        props.mark_queried("ray_origin_radius");
        props.mark_queried("ray_origin_center");
        props.mark_queried("ray_origin_a");
        props.mark_queried("ray_origin_b");
        props.mark_queried("ray_origin_shape");
        props.mark_queried("ray_origin_type");
    }

    /// This sensor does not occupy any particular region of space, return an
    /// invalid bounding box
    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    template <RayTargetType TargetType>
    using Impl = DistantSensorImpl<Float, Spectrum, TargetType>;

    /**
     * Recursively expand into an implementation specialized to the ray origin specification.
     */
    std::vector<ref<Object>> expand() const override {
        ref<Object> result;
        switch (m_ray_target_type) {
            case RayTargetType::Disk:
                result = (Object *) new Impl<RayTargetType::Disk>(m_props);
                break;
            case RayTargetType::Rectangle:
                result = (Object *) new Impl<RayTargetType::Rectangle>(m_props);
                break;
            case RayTargetType::Distant:
                result = (Object *) new Impl<RayTargetType::Distant>(m_props);
                break;
            default:
                Throw("Unsupported ray origin type!");
        }
        return { result };
    }

    MTS_DECLARE_CLASS()

protected:
    Properties m_props;
    RayTargetType m_ray_target_type;
};

template <typename Float, typename Spectrum, RayTargetType TargetType>
class DistantSensorImpl final : public Sensor<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Sensor, m_world_transform, m_film)
    MTS_IMPORT_TYPES(Scene)

    DistantSensorImpl(const Properties &props) : Base(props) {

        if (props.has_property("direction")) {
            if (props.has_property("to_world"))
                Throw("Only one of the parameters 'direction' and 'to_world' "
                      "can be specified at the same time!'");

            ScalarVector3f direction(normalize(props.vector3f("direction")));
            auto [up, unused] = coordinate_system(direction);

            m_world_transform =
                new AnimatedTransform(ScalarTransform4f::look_at(
                    ScalarPoint3f(0.0f), ScalarPoint3f(direction), up));
        }

        if constexpr (TargetType == RayTargetType::Disk) {
            if (!props.has_property("ray_target_center") || !props.has_property("ray_target_radius")) {
                Throw("RayTargetType::Disk requires the 'm_ray_target_center' and 'm_ray_target_radius' parameters");
            }
            m_ray_target_center = props.point3f("ray_target_center");
            m_ray_target_radius = props.float_("ray_target_radius");
        } else if constexpr (TargetType == RayTargetType::Rectangle) {
            if (!props.has_property("ray_target_a") || !props.has_property("ray_target_b")) {
                Throw("RayTargetType::Rectangle requires the 'm_ray_target_a' and 'm_ray_target_b' parameters");
            }
            m_ray_target_a = props.point3f("ray_target_a");
            m_ray_target_b = props.point3f("ray_target_b");
            if (!(m_ray_target_a.z() == m_ray_target_b.z())) {
                Throw("z-components of m_ray_target_a and m_ray_target_b do not match. "
                      "%s, %s", m_ray_target_a, m_ray_target_b);
            }
        } else if constexpr (TargetType == RayTargetType::Distant) {

        } else {
            NotImplementedError("Unsupported RayTargetType");
        }
        
        std::string origin_type = props.string("ray_origin_type", "disk");

        // set ray origin type for ray validity checks
        if (origin_type == "disk") {
            m_ray_origin_type = RayOriginType::Disk;
        } else if (origin_type == "rectangle") {
            m_ray_origin_type = RayOriginType::Rectangle;
        } else if (origin_type == "shape") {
            m_ray_origin_type = RayOriginType::Shape
        } else {
            Throw("Unknown ray origin type: %s", origin_type);
        }

        if (m_ray_origin_type == RayOriginType::Disk) {
            m_ray_origin_center = props.point3f("ray_origin_center");
            m_ray_origin_radius = props.float_("ray_origin_radius");
        } else if (m_ray_origin_type == RayOriginType::Rectangle) {
            m_ray_origin_a      = props.point3f("ray_origin_a");
            m_ray_origin_b      = props.point3f("ray_origin_b");
        } else if (m_ray_origin_type == RayOriginType::Shape) {
            for (auto &[name, obj] : props.objects(false)) {
                ray_origin_shape = dynamic_cast<Shape * >(obj.get());

                if (ray_origin_shape)
                    m_ray_origin_shape = ray_origin_shape;
                    props.mark_queried("ray_origin_shape")
            }
            if (!m_ray_origin_shape){
                Throw("Could not instantiate a shape for ray origins.");
            }
        }


        if (m_film->size() != ScalarPoint2i(1, 1))
            Throw("This sensor only supports films of size 1x1 Pixels!");

        if (m_film->reconstruction_filter()->radius() >
            0.5f + math::RayEpsilon<Float>)
            Log(Warn, "This sensor should be used with a reconstruction filter "
                      "with a radius of 0.5 or lower (e.g. default box)");

        props.mark_queried("ray_target");
        props.mark_queried("direction");
        props.mark_queried("to_world");

    }

    void set_scene(const Scene *scene) override {
        m_bsphere = scene->bbox().bounding_sphere();
        m_bsphere.radius =
            max(math::RayEpsilon<Float>,
                m_bsphere.radius * (1.f + math::RayEpsilon<Float>) );
        if constexpr (TargetType == RayTargetType::Distant) {
            m_ray_target_area = math::Pi<Float> * sqr(m_bsphere.radius);
        } else if constexpr (TargetType == RayTargetType::Disk) {
            m_ray_target_area = math::Pi<Float> * sqr(m_ray_target_radius);
        } else if constexpr (TargetType == RayTargetType::Rectangle) {
            m_ray_target_area = abs(m_ray_target_b.x()-m_ray_target_a.x()) * abs(m_ray_target_b.y()-m_ray_target_a.y());
        } else {
            NotImplementedError("Unsupported RayTargetType");
        }

        // rays must begin outside of the bounding box
        // to ensure that, we trace a ray from the bbox' xy-center
        // and z=0 plane until its z component is larger than 
        // the maximum value of bbox' z component plus a margin of 10 percent
        auto trafo          = m_world_transform->eval(0.f, true);
        Vector3f direction  = -trafo.transform_affine(Vector3f{ 0.f, 0.f, 1.f });
        Float bbox_zmax     = scene->bbox().max.z() * 1.1f;
        m_ray_target_length = bbox_zmax / direction.z();
        if (m_ray_target_length == 0){
            m_ray_target_length = 0.1f;
        }
    }

    bool validate_ray(Point3f ray_target, Vector3f ray_direction) {
        if (m_ray_origin_type == RayTargetType::Disk) {
                Float ray_valid_distance = abs(m_ray_origin_center.z() / ray_direction.z());
                Point3f ray_valid_point = ray_target - ray_direction * ray_valid_distance;
                if (sqrt(sqr(ray_valid_point.x()) + sqr(ray_valid_point.y())) > m_ray_origin_radius){
                    return false
                } else { return true }
        } else if (m_ray_origin_type == RayTargetType::Rectangle) {
            Float ray_valid_distance = abs(m_ray_origin_a.z() / ray_direction.z());
            Point3f ray_valid_point = ray_target - ray_direction * ray_valid_distance;
            if (ray_valid_point.x() < min(m_ray_origin_a.x(), m_ray_origin_b.x()) or
                ray_valid_point.x() > max(m_ray_origin_a.x(), m_ray_origin_b.x()) or
                ray_valid_point.y() < min(m_ray_origin_a.y(), m_ray_origin_b.y()) or
                ray_valid_point.y() > max(m_ray_origin_a.y(), m_ray_origin_b.y())) {
                    return false
                } else { return true }
        } else if (m_ray_origin_type == RayOriginType::Shape) {
            Ray3f test_ray;
            test_ray.time = 0.f;
            test_ray.o = ray_target;
            test_ray.d = -ray_direction;
            if (!m_ray_origin_shape->ray_test(test_ray, true)){
                return false
            } else {return true}
        } else {
            Throw("Unsupported ray origin type: %s", m_ray_origin_type);
            return false
        }
    }
    // won't compile. why? :O

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f & /*film_sample*/,
                                          const Point2f &aperture_sample,
                                          Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);
        Ray3f ray;
        ray.time = time;

        // 1. Sample spectrum
        auto [wavelengths, wav_weight] =
            sample_wavelength<Float, Spectrum>(wavelength_sample);
        ray.wavelengths = wavelengths;

        // 2. Set ray direction
        auto trafo = m_world_transform->eval(time, active);
        ray.d      = trafo.transform_affine(Vector3f{ 0.f, 0.f, 1.f });

        // 3. Sample ray origin
        Spectrum ray_weight = 0.f;

        if constexpr (TargetType == RayTargetType::Distant) {
            // If no ray origin is defined, sample a target point on the
            // bounding sphere's cross section
            Point2f offset =
                warp::square_to_uniform_disk_concentric(aperture_sample);
            Vector3f perp_offset =
                trafo.transform_affine(Vector3f{ offset.x(), offset.y(), 0.f });
            Point3f ray_target = m_bsphere.center + perp_offset * m_bsphere.radius;
            ray.o = ray_target - ray.d * m_ray_target_length;
            ray_weight = wav_weight * m_ray_target_area;

            if (!validate_ray(ray_target, ray.d)) {
                Throw("Invalid ray origin!");
            }
        } 
        if constexpr (TargetType == RayTargetType::Rectangle) {
            Float ray_target_x = abs(m_ray_target_b.x() - m_ray_target_a.x()) * aperture_sample.x() + min(m_ray_target_a.x(), m_ray_target_b.x());
            Float ray_target_y = abs(m_ray_target_b.y() - m_ray_target_a.y()) * aperture_sample.y() + min(m_ray_target_a.y(), m_ray_target_b.y());
            Point3f ray_target = Point3f{ray_target_x, ray_target_y, m_ray_target_a.z()};
            ray.o = ray_target - ray.d * m_ray_target_length;
            ray_weight = wav_weight * m_ray_target_area * Frame3f::cos_theta(-ray.d);

            if (!validate_ray(ray_target, ray.d)) {
                Throw("Invalid ray origin!");
            }
        }   
        if constexpr (TargetType == RayTargetType::Disk) {
            Point2f disk_sample = Point2f(warp::square_to_uniform_disk(aperture_sample)) * m_ray_target_radius;
            Point3f sample_disk = Point3f(disk_sample.x(), disk_sample.y(), 0.f) + m_ray_target_center;
            Point3f ray_target = Point3f{sample_disk.x(), sample_disk.y(), m_ray_target_center.z()};
            ray.o = ray_target - ray.d * m_ray_target_length;
            ray_weight = wav_weight * 1 * Frame3f::cos_theta(-ray.d);
            
            if (!validate_ray(ray_target, ray.d)) {
                Throw("Invalid ray origin!");
            }
        }

        ray.update();
        return {ray, ray_weight};
    }

    std::pair<RayDifferential3f, Spectrum> sample_ray_differential(
        Float time, Float wavelength_sample, const Point2f & /*film_sample*/,
        const Point2f &aperture_sample, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);
        RayDifferential3f ray;
        ray.time = time;

        // 1. Sample spectrum
        auto [wavelengths, wav_weight] =
            sample_wavelength<Float, Spectrum>(wavelength_sample);
        ray.wavelengths = wavelengths;

        // 2. Set ray direction
        auto trafo = m_world_transform->eval(time, active);
        ray.d      = trafo.transform_affine(Vector3f{ 0.f, 0.f, 1.f });

        // 3. Sample ray origin
        Spectrum ray_weight = 0.f;

        if constexpr (TargetType == RayTargetType::Distant) {
            // If no ray origin is defined, sample a target point on the
            // bounding sphere's cross section
            Point2f offset =
                warp::square_to_uniform_disk_concentric(aperture_sample);
            Vector3f perp_offset =
                trafo.transform_affine(Vector3f{ offset.x(), offset.y(), 0.f });
            Point3f ray_target = m_bsphere.center + perp_offset * m_bsphere.radius;
            ray.o = ray_target - ray.d * m_ray_target_length;
            ray_weight = wav_weight * m_ray_target_area;

            if (!validate_ray(ray_target, ray.d)) {
                Throw("Invalid ray origin!");
            }
        } 
        if constexpr (TargetType == RayTargetType::Rectangle) {
            Float ray_target_x = abs(m_ray_target_b.x() - m_ray_target_a.x()) * aperture_sample.x() + min(m_ray_target_a.x(), m_ray_target_b.x());
            Float ray_target_y = abs(m_ray_target_b.y() - m_ray_target_a.y()) * aperture_sample.y() + min(m_ray_target_a.y(), m_ray_target_b.y());
            Point3f ray_target = Point3f{ray_target_x, ray_target_y, m_ray_target_a.z()};
            ray.o = ray_target - ray.d * m_ray_target_length;
            ray_weight = wav_weight * m_ray_target_area * Frame3f::cos_theta(-ray.d);

            if (!validate_ray(ray_target, ray.d)) {
                Throw("Invalid ray origin!");
            }
        }
        if constexpr (TargetType == RayTargetType::Disk) {
            Point2f disk_sample = Point2f(warp::square_to_uniform_disk(aperture_sample)) * m_ray_target_radius;
            Point3f ray_target = Point3f(disk_sample.x(), disk_sample.y(), 0.f) + m_ray_target_center;
            ray.o = ray_target - ray.d * m_ray_target_length;
            ray_weight = wav_weight * 1 * Frame3f::cos_theta(-ray.d);

            if (!validate_ray(ray_target, ray.d)) {
                Throw("Invalid ray origin!");
            }
        }
        
        // 4. Set differentials; since the film size is always 1x1, we don't
        //    have differentials
        ray.has_differentials = false;

        ray.update();
        return {ray, ray_weight};
    }

    /// This sensor does not occupy any particular region of space, return an
    /// invalid bounding box
    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    std::string to_string() const override {
        std::ostringstream oss;
        if constexpr (TargetType == RayTargetType::Rectangle) {
            oss << "DistantSensor[" << std::endl
                << "  world_transform = " << m_world_transform << "," << std::endl
                << "  ray_target_type = " << "rectangle" << "," << std::endl
                << "  ray_target_point_a = " << m_ray_target_a << "," << std::endl
                << "  ray_target_point_b = " << m_ray_target_b << "," << std::endl
                << "  film = " << m_film << "," << std::endl
                << "]";
            return oss.str();
        } else if constexpr (TargetType == RayTargetType::Disk) {
            oss << "DistantSensor[" << std::endl
                << "  world_transform = " << m_world_transform << "," << std::endl
                << "  ray_target_type = " << "disk" << "," << std::endl
                << "  ray_target_center = " << m_ray_target_center << "," << std::endl
                << "  ray_target_point_radius = " << m_ray_target_radius << "," << std::endl
                << "  film = " << m_film << "," << std::endl
                << "]";
            return oss.str();
        } else if constexpr (TargetType == RayTargetType::Distant) {
            oss << "DistantSensor[" << std::endl
                << "  world_transform = " << m_world_transform << "," << std::endl
                << "  ray_target_type = " << "distant" << "," << std::endl
                << "  bsphere = " << m_bsphere << "," << std::endl
                << "  film = " << m_film << "," << std::endl
                << "]";
            return oss.str();
        } else {
            Throw("Unknown ray origin type");
        }
    }

    MTS_DECLARE_CLASS()

protected:
    ScalarBoundingSphere3f m_bsphere;
    Point3f m_ray_target_center;
    Float m_ray_target_radius;
    Point3f m_ray_target_a;
    Point3f m_ray_target_b;
    Float m_ray_target_area;
    Float m_ray_target_length;

    
    Point3f m_ray_origin_center;
    Float m_ray_origin_radius;
    Point3f m_ray_origin_a;
    Point3f m_ray_origin_b;
    RayTargetType m_ray_origin_type;
    Shape *m_origin_shape;
};

MTS_IMPLEMENT_CLASS_VARIANT(DistantSensor, Sensor)
MTS_EXPORT_PLUGIN(DistantSensor, "DistantSensor")

NAMESPACE_BEGIN(detail)
template <RayTargetType TargetType> constexpr const char *distant_sensor_class_name() {
    if constexpr (TargetType == RayTargetType::Disk) {
        return "DistantSensor_Disk";
    } else if constexpr (TargetType == RayTargetType::Rectangle) {
        return "DistantSensor_Rectangle";
    } else if constexpr (TargetType == RayTargetType::Distant) {
        return "DistantSensor_Distant";
    }
}
NAMESPACE_END(detail)

template <typename Float, typename Spectrum, RayTargetType TargetType>
Class *DistantSensorImpl<Float, Spectrum, TargetType>::m_class = new Class(
    detail::distant_sensor_class_name<TargetType>(), "Sensor",
    ::mitsuba::detail::get_variant<Float, Spectrum>(), nullptr, nullptr);

template <typename Float, typename Spectrum, RayTargetType TargetType>
const Class *DistantSensorImpl<Float, Spectrum, TargetType>::class_() const {
    return m_class;
}

NAMESPACE_END(mitsuba)