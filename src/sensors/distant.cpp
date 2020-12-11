#include <mitsuba/core/bbox.h>
#include <mitsuba/core/bsphere.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sensor.h>

NAMESPACE_BEGIN(mitsuba)

enum class RayOriginType {Distant, Rectangle, Disk};

// Forward declaration of specialized DistantSensor
template <typename Float, typename Spectrum, RayOriginType OriginType>
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
 * - origin
   - |string|
   - *Optional.* Specify the ray origin sampling strategy. 
     If set to ``"rectangle"``, ray origins will be sampled uniformly from 
     a xy-plane-aligned rectangular surface defined by the parameters 
     ``origin_min`` and `` origin_max``. If set to ``"disk"``, ray origins will
     be sampled uniformly from a xy-plane-aligned disk defined by the parameters 
     ``origin_center`` and ``origin_radius``. If set to ``"distant"``, ray origins 
     will be sampled to uniformly cover the entire scene and will be positioned 
     on a bounding sphere. Default: ``"distant"``.
 * - origin_a
   - |point|
   - Coordinates in the xy-plane for the rectangular origin area.
 * - origin_b
   - |point|
   - Maximum coordinates in the xy-plane for the rectangular origin area.
 * - origin_center
   - |point|
   - Center point in the xy-plane for the circular origin area.
 * - origin_radius
   - |float|
   - Radius for the circular origin area
    

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

If the ``origin`` parameter is set, the sensor looks at a single point and
records a (spectral) radiant flux per unit surface area per unit solid angle 
(in unit power per unit surface area per unit solid angle).

With the ``origin_`` parameters users can define an area parallel to the xy-plane
that the sensor's rays will originate from. The area can be defined in two ways:

- Setting `origin` to 'rectangle' lets users define a rectangular area through 
  two points `origin_a` and `origin_b`. The resulting area will cover the area 
  from the smaller to the larger value in both dimensions.
- Setting `origin` to 'circle'  lets users define a circular origin area 
  through a center `origin_center` and a radius `origin_radius`.
- If `origin` is not set, the sensor will origin the entire cross section
  of the scene's bounding sphere.

*/

template <typename Float, typename Spectrum>
class DistantSensor final : public Sensor<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Sensor, m_world_transform, m_film)
    MTS_IMPORT_TYPES()

    DistantSensor(const Properties &props) : Base(props), m_props(props) {

        std::string origin;
        if (props.has_property("origin")) {
            origin = props.string("origin", "distant");
        } else {
            origin = std::string("distant");
        }

        if (origin == "rectangle") {
            m_origin_type = RayOriginType::Rectangle;
        } else if (origin == "disk") {
            m_origin_type = RayOriginType::Disk;
        } else if (origin == "distant") {
            m_origin_type = RayOriginType::Distant;
        }

        props.mark_queried("origin_center");
        props.mark_queried("origin_radius");
        props.mark_queried("origin_a");
        props.mark_queried("origin_b");
        props.mark_queried("direction");
        props.mark_queried("to_world");
    }

    /// This sensor does not occupy any particular region of space, return an
    /// invalid bounding box
    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    template <RayOriginType OriginType>
    using Impl = DistantSensorImpl<Float, Spectrum, OriginType>;

    /**
     * Recursively expand into an implementation specialized to the origin specification.
     */
    std::vector<ref<Object>> expand() const override {
        ref<Object> result;
        switch (m_origin_type) {
            case RayOriginType::Disk:
                result = (Object *) new Impl<RayOriginType::Disk>(m_props);
                break;
            case RayOriginType::Rectangle:
                result = (Object *) new Impl<RayOriginType::Rectangle>(m_props);
                break;
            case RayOriginType::Distant:
                result = (Object *) new Impl<RayOriginType::Distant>(m_props);
                break;
            default:
                Throw("Unsupported ray origin type!");
        }
        return { result };
    }

    MTS_DECLARE_CLASS()

protected:
    Properties m_props;
    RayOriginType m_origin_type;
};

template <typename Float, typename Spectrum, RayOriginType OriginType>
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

        if constexpr (OriginType == RayOriginType::Disk) {
            if (!props.has_property("origin_center") || !props.has_property("origin_radius")) {
                Throw("Circular origin requires the 'origin_center' and 'origin_radius' parameters");
            }
            m_origin_center = props.point3f("origin_center");
            m_origin_radius = props.float_("origin_radius");
            m_origin_area = math::Pi<Float> * sqr(m_origin_radius);
        } else if constexpr (OriginType == RayOriginType::Rectangle) {
            if (!props.has_property("origin_a") || !props.has_property("origin_b")) {
                Throw("Rectangular origin requires the 'origin_a' and 'origin_b' parameters");
            }
            if (!(m_origin_a.z() == m_origin_b.z())) {
                Throw("z-components of origin_a and origin_b do not match. "
                      "Cannot determine origin zone elevation.");
            }
            m_origin_a = props.point3f("origin_a");
            m_origin_b = props.point3f("origin_b");
            m_origin_area = abs(m_origin_b.x()-m_origin_a.x()) * abs(m_origin_b.y()-m_origin_a.y());
        } else if constexpr (OriginType == RayOriginType::Distant) {
            m_origin_area = 0;
        }

        if (m_film->size() != ScalarPoint2i(1, 1))
            Throw("This sensor only supports films of size 1x1 Pixels!");

        if (m_film->reconstruction_filter()->radius() >
            0.5f + math::RayEpsilon<Float>)
            Log(Warn, "This sensor should be used with a reconstruction filter "
                      "with a radius of 0.5 or lower (e.g. default box)");

        props.mark_queried("origin");
        props.mark_queried("direction");
        props.mark_queried("to_world");

    }


    void set_scene(const Scene *scene) override {
        m_bsphere = scene->bbox().bounding_sphere();
        m_bsphere.radius =
            max(math::RayEpsilon<Float>,
                m_bsphere.radius * (1.f + math::RayEpsilon<Float>) );
    }

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
        if constexpr (OriginType == RayOriginType::Distant) {
            // If no origin is defined, sample a target point on the
            // bounding sphere's cross section
            Point2f offset =
                warp::square_to_uniform_disk_concentric(aperture_sample);
            Vector3f perp_offset =
                trafo.transform_affine(Vector3f{ offset.x(), offset.y(), 0.f });
            ray.o = m_bsphere.center + (perp_offset - ray.d) * m_bsphere.radius;
        } 
        if constexpr (OriginType == RayOriginType::Rectangle) {
            Float origin_x = abs(m_origin_b.x() - m_origin_a.x()) * aperture_sample.x() + min(m_origin_a.x(), m_origin_b.x());
            Float origin_y = abs(m_origin_b.y() - m_origin_a.y()) * aperture_sample.y() + min(m_origin_a.y(), m_origin_b.y());
            ray.o = Point3f{origin_x, origin_y, m_origin_a.z()};
        }
        if constexpr (OriginType == RayOriginType::Disk) {
            Point2f disk_sample = Point2f(warp::square_to_uniform_disk(aperture_sample)) * m_origin_radius;
            Point3f sample_disk = Point3f(disk_sample.x(), disk_sample.y(), 0.f) + m_origin_center;
            ray.o = Point3f{sample_disk.x(), sample_disk.y(), m_origin_center.z()};
        }

        ray.update();
        return std::make_pair(
            ray, wav_weight * m_origin_area * Frame3f::cos_theta(-ray.d));
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
        Float origin_area;
        if constexpr (OriginType == RayOriginType::Distant) {
            // If no origin is defined, sample a target point on the
            // bounding sphere's cross section
            Point2f offset =
                warp::square_to_uniform_disk_concentric(aperture_sample);
            Vector3f perp_offset =
                trafo.transform_affine(Vector3f{ offset.x(), offset.y(), 0.f });
            ray.o = m_bsphere.center + (perp_offset - ray.d) * m_bsphere.radius;
            origin_area = math::Pi<Float> * sqr(m_bsphere.radius);
        } 
        if constexpr (OriginType == RayOriginType::Rectangle) {
            Float origin_x = abs(m_origin_b.x() - m_origin_a.x()) * aperture_sample.x() + min(m_origin_a.x(), m_origin_b.x());
            Float origin_y = abs(m_origin_b.y() - m_origin_a.y()) * aperture_sample.y() + min(m_origin_a.y(), m_origin_b.y());
            ray.o = Point3f{origin_x, origin_y, m_origin_a.z()};
            origin_area = m_origin_area;
        }
        if constexpr (OriginType == RayOriginType::Disk) {
            Point2f disk_sample = Point2f(warp::square_to_uniform_disk(aperture_sample)) * m_origin_radius;
            Point3f sample_disk = Point3f(disk_sample.x(), disk_sample.y(), 0.f) + m_origin_center;
            ray.o = Point3f{sample_disk.x(), sample_disk.y(), m_origin_center.z()};
            origin_area = m_origin_area;
        }

        // 4. Set differentials; since the film size is always 1x1, we don't
        //    have differentials
        ray.has_differentials = false;

        ray.update();
        return std::make_pair(
            ray, wav_weight * origin_area * Frame3f::cos_theta(-ray.d));
    }

    /// This sensor does not occupy any particular region of space, return an
    /// invalid bounding box
    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    std::string to_string() const override {
        std::ostringstream oss;
        if constexpr (OriginType == RayOriginType::Rectangle) {
            oss << "DistantSensor[" << std::endl
                << "  world_transform = " << m_world_transform << "," << std::endl
                << "  origin_type = " << "rectangle" << "," << std::endl
                << "  origin_point_a = " << m_origin_a << "," << std::endl
                << "  origin_point_b = " << m_origin_b << "," << std::endl
                << "  bsphere = " << m_bsphere << "," << std::endl
                << "  film = " << m_film << "," << std::endl
                << "]";
            return oss.str();
        } else if constexpr (OriginType == RayOriginType::Disk) {
            oss << "DistantSensor[" << std::endl
                << "  world_transform = " << m_world_transform << "," << std::endl
                << "  origin_type = " << "disk" << "," << std::endl
                << "  origin_center = " << m_origin_center << "," << std::endl
                << "  origin_point_radius = " << m_origin_radius << "," << std::endl
                << "  bsphere = " << m_bsphere << "," << std::endl
                << "  film = " << m_film << "," << std::endl
                << "]";
            return oss.str();
        } else if constexpr (OriginType == RayOriginType::Distant) {
            oss << "DistantSensor[" << std::endl
                << "  world_transform = " << m_world_transform << "," << std::endl
                << "  origin_type = " << "distant" << "," << std::endl
                << "  bsphere = " << m_bsphere << "," << std::endl
                << "  film = " << m_film << "," << std::endl
                << "]";
            return oss.str();
        } else {
            Throw("Unknown origin type");
        }
    }

    MTS_DECLARE_CLASS()

protected:
    ScalarBoundingSphere3f m_bsphere;
    Point3f m_origin_center;
    Float m_origin_radius;
    Point3f m_origin_a;
    Point3f m_origin_b;
    Float m_origin_area;
};

MTS_IMPLEMENT_CLASS_VARIANT(DistantSensor, Sensor)
MTS_EXPORT_PLUGIN(DistantSensor, "DistantSensor")

NAMESPACE_BEGIN(detail)
template <RayOriginType OriginType> constexpr const char *distant_sensor_class_name() {
    if constexpr (OriginType == RayOriginType::Disk) {
        return "DistantSensor_Disk";
    } else if constexpr (OriginType == RayOriginType::Rectangle) {
        return "DistantSensor_Rectangle";
    } else if constexpr (OriginType == RayOriginType::Distant) {
        return "DistantSensor_Distant";
    }
}
NAMESPACE_END(detail)

template <typename Float, typename Spectrum, RayOriginType OriginType>
Class *DistantSensorImpl<Float, Spectrum, OriginType>::m_class = new Class(
    detail::distant_sensor_class_name<OriginType>(), "Sensor",
    ::mitsuba::detail::get_variant<Float, Spectrum>(), nullptr, nullptr);

template <typename Float, typename Spectrum, RayOriginType OriginType>
const Class *DistantSensorImpl<Float, Spectrum, OriginType>::class_() const {
    return m_class;
}

NAMESPACE_END(mitsuba)