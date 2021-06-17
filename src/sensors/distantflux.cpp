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

// Forward declaration of specialized DistantSensor
template <typename Float, typename Spectrum, RayTargetType TargetType>
class DistantFluxSensorImpl;

/**!

.. _sensor-distant:

Distant fluxmeter sensor (:monosp:`distantflux`)
------------------------------------------------

.. pluginparameters::

 * - to_world
   - |transform|
   - Sensor-to-world transformation matrix.
 * - target
   - |point| or nested :paramtype:`shape` plugin
   - *Optional.* Define the ray target sampling strategy.
     If this parameter is unset, ray target points are sampled uniformly on
     the cross section of the scene's bounding sphere.
     If a |point| is passed, rays will target it.
     If a shape plugin is passed, ray target points will be sampled from its
     surface.

This sensor plugin implements a distant sensor which records the radiative flux
density leaving the scene (in W/m², scaled by scene unit length). It covers a
hemisphere defined by its ``to_world`` parameter and mapped to film coordinates.

The ``to_world`` transform is best set using a
:py:meth:`~mitsuba.core.Transform4f.look_at`. The default orientation covers a
hemisphere defined by the [0, 0, 1] direction, and the ``up`` film direction is
set to [0, 1, 0].

Using a 1x1 film with a stratified sampler is recommended.
A different film size can also be used. In that case, the exitant flux is given
by the mean of all pixel values.

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
*/

template <typename Float, typename Spectrum>
class DistantFluxSensor final : public Sensor<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Sensor)
    MTS_IMPORT_TYPES(Scene, Shape)

    DistantFluxSensor(const Properties &props) : Base(props), m_props(props) {

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

        props.mark_queried("to_world");
        props.mark_queried("target");
    }

    // This must be implemented. However, it won't be used in practice:
    // instead, DistantFluxSensorImpl::bbox() is used when the plugin is
    // instantiated.
    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    template <RayTargetType TargetType>
    using Impl = DistantFluxSensorImpl<Float, Spectrum, TargetType>;

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
class DistantFluxSensorImpl final : public Sensor<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Sensor, m_world_transform, m_film)
    MTS_IMPORT_TYPES(Scene, Shape)

    DistantFluxSensorImpl(const Properties &props) : Base(props) {
        // Check reconstruction filter radius
        if (m_film->reconstruction_filter()->radius() >
            0.5f + math::RayEpsilon<Float>) {
            Log(Warn, "This sensor is best used with a reconstruction filter "
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

        // Set reference surface normal (in world coords)
        ScalarTransform4f trafo = props.transform("to_world", ScalarTransform4f());
        m_reference_normal = trafo.transform_affine(ScalarVector3f{0.f, 0.f, 1.f});
    }

    void set_scene(const Scene *scene) override {
        m_bsphere = scene->bbox().bounding_sphere();
        m_bsphere.radius =
            max(math::RayEpsilon<Float>,
                m_bsphere.radius * (1.f + math::RayEpsilon<Float>));
    }

    std::pair<Ray3f, Spectrum>
    sample_ray_impl(Float time, Float wavelength_sample,
               const Point2f &film_sample,
               const Point2f &aperture_sample, Mask active) const {
        MTS_MASK_ARGUMENT(active);

        Ray3f ray;
        ray.time = time;

        // Sample spectrum
        auto [wavelengths, wav_weight] =
            sample_wavelength<Float, Spectrum>(wavelength_sample);
        ray.wavelengths = wavelengths;

        // Sample ray direction
        auto trafo = m_world_transform->eval(time, active);
        ray.d = -trafo.transform_affine(
            warp::square_to_uniform_hemisphere(film_sample)
        );

        // Sample target point and position ray origin
        Spectrum ray_weight =
            dot(-ray.d, m_reference_normal) / warp::square_to_uniform_hemisphere_pdf(ray.d);

        if constexpr (TargetType == RayTargetType::Point) {
            ray.o = m_target_point - 2.f * ray.d * m_bsphere.radius;
            ray_weight *= wav_weight;
        } else if constexpr (TargetType == RayTargetType::Shape) {
            // Use area-based sampling of shape
            PositionSample3f ps =
                m_target_shape->sample_position(time, aperture_sample, active);
            ray.o = ps.p - 2.f * ray.d * m_bsphere.radius;
            ray_weight *= wav_weight / (ps.pdf * m_target_shape->surface_area());
        } else { // if constexpr (TargetType == RayTargetType::None) {
            // Sample target uniformly on bounding sphere cross section defined
            // by reference surface normal
            Point2f offset =
                warp::square_to_uniform_disk_concentric(aperture_sample);
            Vector3f perp_offset =
                trafo.transform_affine(Vector3f(offset.x(), offset.y(), 0.f));
            ray.o = m_bsphere.center + (perp_offset - ray.d) * m_bsphere.radius;
            ray_weight *= wav_weight;
        }

        return { ray, ray_weight & active };
    }

    std::pair<Ray3f, Spectrum>
    sample_ray(Float time, Float wavelength_sample,
               const Point2f &film_sample,
               const Point2f &aperture_sample, Mask active) const override {
        MTS_MASK_ARGUMENT(active);

        auto [ray, ray_weight] = sample_ray_impl(
            time, wavelength_sample, film_sample, aperture_sample, active);

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

        // Since the film size is always 1x1, we don't have differentials
        ray.has_differentials = false;

        ray.update();
        return { ray, ray_weight & active };
    }

    // This sensor does not occupy any particular region of space, return an
    // invalid bounding box
    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "DistantFluxSensor[" << std::endl
            << "  reference_normal = " << m_reference_normal << "," << std::endl
            << "  world_transform = " << string::indent(m_world_transform) << "," << std::endl
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
    Vector3f m_reference_normal;
};

MTS_IMPLEMENT_CLASS_VARIANT(DistantFluxSensor, Sensor)
MTS_EXPORT_PLUGIN(DistantFluxSensor, "DistantFluxSensor")

NAMESPACE_BEGIN(detail)
template <RayTargetType TargetType>
constexpr const char *distant_sensor_class_name() {
    if constexpr (TargetType == RayTargetType::Shape) {
        return "DistantFluxSensor_Shape";
    } else if constexpr (TargetType == RayTargetType::Point) {
        return "DistantFluxSensor_Point";
    } else if constexpr (TargetType == RayTargetType::None) {
        return "DistantFluxSensor_NoTarget";
    }
}
NAMESPACE_END(detail)

template <typename Float, typename Spectrum, RayTargetType TargetType>
Class *DistantFluxSensorImpl<Float, Spectrum, TargetType>::m_class = new Class(
    detail::distant_sensor_class_name<TargetType>(), "Sensor",
    ::mitsuba::detail::get_variant<Float, Spectrum>(), nullptr, nullptr);

template <typename Float, typename Spectrum, RayTargetType TargetType>
const Class *DistantFluxSensorImpl<Float, Spectrum, TargetType>::class_() const {
    return m_class;
}

NAMESPACE_END(mitsuba)
