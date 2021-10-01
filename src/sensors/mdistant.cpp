#include <mitsuba/core/bbox.h>
#include <mitsuba/core/bsphere.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sensor.h>
#include <mitsuba/render/shape.h>

NAMESPACE_BEGIN(mitsuba)

enum class RayTargetType { Shape, Point, None };

// Forward declaration of specialized MultiDistantSensor
template <typename Float, typename Spectrum, RayTargetType TargetType>
class MultiDistantSensorImpl;

/**!

.. _sensor-mdistant:

Multi distant radiance meter (:monosp:`mdistant`)
-------------------------------------------------

.. pluginparameters::

 * - directions
   - |string|
   - Comma-separated list of directions in which the sensors are pointing in
     world coordinates.

 * - target
   - |point| or nested :paramtype:`shape` plugin
   - *Optional.* Define the ray target sampling strategy.
     If this parameter is unset, ray target points are sampled uniformly on
     the cross section of the scene's bounding sphere.
     If a |point| is passed, rays will target it.
     If a shape plugin is passed, ray target points will be sampled from its
     surface.

This sensor plugin aggregates an arbitrary number of distant directional sensors
which records the spectral radiance leaving the scene in specified directions.
It is the aggregation of multiple :monosp:`distant` sensors.

By default, ray target points are sampled from the cross section of the scene's
bounding sphere. The ``target`` parameter can be set to restrict ray target
sampling to a specific subregion of the scene. The recorded radiance is averaged
over the targeted geometry.

Ray origins are positioned outside of the scene's geometry.

.. warning::

   If this sensor is used with a targeting strategy leading to rays not hitting
   the scene's geometry (*e.g.* default targeting strategy), it will pick up
   ambient emitter radiance samples (or zero values if no ambient emitter is
   defined). Therefore, it is almost always preferrable to use a nondefault
   targeting strategy.

.. important::

   This sensor must be used with a film with size (`N`, 1), where `N` is the
   number of aggregated sensors, and is best used with a default :monosp:`box`
   reconstruction filter.
*/

template <typename Float, typename Spectrum>
class MultiDistantSensor final : public Sensor<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Sensor, m_world_transform, m_film)
    MTS_IMPORT_TYPES(Scene, Shape)

    MultiDistantSensor(const Properties &props) : Base(props), m_props(props) {

        if (props.has_property("to_world")) {
            Throw("This sensor is specified through a set of origin and "
                  "direction values and cannot use the to_world transform.");
        }

        // Get target
        if (props.has_property("target")) {
            if (props.type("target") == Properties::Type::Array3f) {
                props.point3f("target");
                m_target_type = RayTargetType::Point;
            } else if (props.type("target") == Properties::Type::Object) {
                // We assume it's a shape
                m_target_type = RayTargetType::Shape;
            } else {
                Throw("Unsupported 'target' parameter type");
            }
        } else {
            m_target_type = RayTargetType::None;
        }

        props.mark_queried("directions");
        props.mark_queried("target");
    }

    // This must be implemented. However, it won't be used in practice:
    // instead, MultiDistantSensorImpl::bbox() is used when the plugin is
    // instantiated.
    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    template <RayTargetType TargetType>
    using Impl = MultiDistantSensorImpl<Float, Spectrum, TargetType>;

    // Recursively expand into an implementation specialized to the target
    // specification.
    std::vector<ref<Object>> expand() const override {
        ref<Object> result;
        switch (m_target_type) {
            case RayTargetType::Shape:
                result = (Object *) new Impl<RayTargetType::Shape>(m_props);
                break;
            case RayTargetType::Point:
                result = (Object *) new Impl<RayTargetType::Point>(m_props);
                break;
            case RayTargetType::None:
                result = (Object *) new Impl<RayTargetType::None>(m_props);
                break;
            default:
                Throw("Unsupported ray target type!");
        }
        return { result };
    }

    MTS_DECLARE_CLASS()

protected:
    Properties m_props;
    RayTargetType m_target_type;
};

template <typename Float, typename Spectrum, RayTargetType TargetType>
class MultiDistantSensorImpl final : public Sensor<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Sensor, m_world_transform, m_film, m_needs_sample_2,
                    m_needs_sample_3)
    MTS_IMPORT_TYPES(Scene, Shape)

    using Matrix           = enoki::Matrix<Float, Transform4f::Size>;
    using TransformStorage = DynamicBuffer<Matrix>;
    using Index            = int32_array_t<Float>;

    MultiDistantSensorImpl(const Properties &props) : Base(props) {
        // Collect directions and set transforms accordingly
        std::vector<std::string> directions_str =
            string::tokenize(props.string("directions"), " ,");

        if (directions_str.size() % 3 != 0)
            Throw("Invalid specification! Number of parameters %s, is not a "
                  "multiple of three.",
                  directions_str.size());

        m_sensor_count = directions_str.size() / 3;
        m_transforms   = empty<TransformStorage>(m_sensor_count * 16);
        m_transforms.managed();

        for (size_t i = 0; i < m_sensor_count; ++i) {
            size_t index = i * 3;
            ScalarVector3f direction =
                ScalarVector3f(std::stof(directions_str[index]),
                               std::stof(directions_str[index + 1]),
                               std::stof(directions_str[index + 2]));

            ScalarVector3f up;
            std::tie(std::ignore, up) = coordinate_system(direction);

            auto transform =
                ScalarTransform4f::look_at(ScalarPoint3f{ 0.f, 0.f, 0.f },
                                           ScalarPoint3f(direction), up)
                    .matrix;
            memcpy(&m_transforms[i * 16], &transform, sizeof(ScalarFloat) * 16);
        }

        // Check film size
        ScalarPoint2i expected_size{ m_sensor_count, 1 };
        if (m_film->size() != expected_size)
            Throw("Film size must be [sensor_count, 1]. Expected %s, "
                  "got %s",
                  expected_size, m_film->size());

        // Check reconstruction filter radius
        if (m_film->reconstruction_filter()->radius() >
            0.5f + math::RayEpsilon<Float>) {
            Log(Warn, "This sensor should be used with a reconstruction filter "
                      "with a radius of 0.5 or lower (e.g. default box)");
        }

        // Set ray target if relevant
        if constexpr (TargetType == RayTargetType::Point) {
            m_target_point = props.point3f("target");
        } else if constexpr (TargetType == RayTargetType::Shape) {
            auto obj       = props.object("target");
            m_target_shape = dynamic_cast<Shape *>(obj.get());

            if (!m_target_shape)
                Throw(
                    "Invalid parameter target, must be a Point3f or a Shape.");
        } else {
            Log(Debug, "No target specified.");
        }

        m_needs_sample_2 = true;
        m_needs_sample_3 = true;
    }

    void set_scene(const Scene *scene) override {
        m_bsphere = scene->bbox().bounding_sphere();
        m_bsphere.radius =
            max(math::RayEpsilon<Float>,
                m_bsphere.radius * (1.f + math::RayEpsilon<Float>) );
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &film_sample,
                                          const Point2f &aperture_sample,
                                          Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        Ray3f ray;
        ray.time = time;

        // Sample spectrum
        auto [wavelengths, wav_weight] =
            sample_wavelength<Float, Spectrum>(wavelength_sample);
        ray.wavelengths = wavelengths;

        // Select sub-sensor
        Int32 sensor_index(film_sample.x() * m_sensor_count);
        Index index(sensor_index);

        Matrix coefficients = gather<Matrix>(m_transforms, index);
        Transform4f trafo(coefficients);

        // Set ray direction
        ray.d = trafo.transform_affine(Vector3f{ 0.f, 0.f, 1.f });

        // Sample target point and position ray origin
        Spectrum ray_weight = 0.f;

        if constexpr (TargetType == RayTargetType::Point) {
            ray.o      = m_target_point - 2.f * ray.d * m_bsphere.radius;
            ray_weight = wav_weight;
        } else if constexpr (TargetType == RayTargetType::Shape) {
            // Use area-based sampling of shape
            PositionSample3f ps =
                m_target_shape->sample_position(time, aperture_sample, active);
            ray.o      = ps.p - 2.f * ray.d * m_bsphere.radius;
            ray_weight = wav_weight / (ps.pdf * m_target_shape->surface_area());
        } else { // if constexpr (TargetType == RayTargetType::None) {
            // Sample target uniformly on bounding sphere cross section
            Point2f offset =
                warp::square_to_uniform_disk_concentric(aperture_sample);
            Vector3f perp_offset =
                trafo.transform_affine(Vector3f{ offset.x(), offset.y(), 0.f });
            ray.o = m_bsphere.center + perp_offset * m_bsphere.radius -
                    ray.d * m_bsphere.radius;
            ray_weight = wav_weight;
        }

        ray.update();

        return { ray, ray_weight & active };
    }

    std::pair<RayDifferential3f, Spectrum> sample_ray_differential(
        Float time, Float wavelength_sample, const Point2f &film_sample,
        const Point2f &aperture_sample, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        RayDifferential3f ray;
        Spectrum ray_weight;

        std::tie(ray, ray_weight) = sample_ray(
            time, wavelength_sample, film_sample, aperture_sample, active);

        // Film pixels are actually independent, we don't have differentials
        ray.has_differentials = false;

        return { ray, ray_weight & active };
    }

    // This sensor does not occupy any particular region of space, return an
    // invalid bounding box
    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "MultiDistantSensor[" << std::endl
            << "  transforms = " << string::indent(m_transforms) << "," << std::endl
            << "  film = " << string::indent(m_film) << "," << std::endl;

        if constexpr (TargetType == RayTargetType::Point)
            oss << "  target = " << m_target_point << std::endl;
        else if constexpr (TargetType == RayTargetType::Shape)
            oss << "  target = " << string::indent(m_target_shape) << std::endl;
        else // if constexpr (TargetType == RayTargetType::None)
            oss << "  target = none" << std::endl;

        oss << "]";

        return oss.str();
    }

    MTS_DECLARE_CLASS()

protected:
    ScalarBoundingSphere3f m_bsphere;
    ref<Shape> m_target_shape;
    Point3f m_target_point;
    TransformStorage m_transforms;
    size_t m_sensor_count;
};

MTS_IMPLEMENT_CLASS_VARIANT(MultiDistantSensor, Sensor)
MTS_EXPORT_PLUGIN(MultiDistantSensor, "MultiDistantSensor")

NAMESPACE_BEGIN(detail)
template <RayTargetType TargetType>
constexpr const char *distant_sensor_class_name() {
    if constexpr (TargetType == RayTargetType::Shape) {
        return "MultiDistantSensor_Shape";
    } else if constexpr (TargetType == RayTargetType::Point) {
        return "MultiDistantSensor_Point";
    } else if constexpr (TargetType == RayTargetType::None) {
        return "MultiDistantSensor_NoTarget";
    }
}
NAMESPACE_END(detail)

template <typename Float, typename Spectrum, RayTargetType TargetType>
Class *MultiDistantSensorImpl<Float, Spectrum, TargetType>::m_class = new Class(
    detail::distant_sensor_class_name<TargetType>(), "Sensor",
    ::mitsuba::detail::get_variant<Float, Spectrum>(), nullptr, nullptr);

template <typename Float, typename Spectrum, RayTargetType TargetType>
const Class *
MultiDistantSensorImpl<Float, Spectrum, TargetType>::class_() const {
    return m_class;
}

NAMESPACE_END(mitsuba)
