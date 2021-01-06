#include <mitsuba/core/properties.h>
#include <mitsuba/render/texture.h>

#include "volume_data.h"

NAMESPACE_BEGIN(mitsuba)

/**!

.. _volume-gridvolume_spectral:

Spectral grid (:monosp:`gridvolume_spectral`)
---------------------------------------------

This plugin implements a gridded volume data source using an arbitrary regular
spectrum. It should be preferred over the ``gridvolume`` plugin when proper
spectral data is to be used.

In practice, this plugin can be seen as a combination of the ``gridvolume``
and ``regular`` plugins. Evaluations outside of the covered spectral range will 
return 0.

WARNING: This plugin is currently in alpha stage and can be subject to deep 
changes.

.. pluginparameters::

 * - filename
   - |string|
   - Filename of the volume data file to be loaded

 * - use_grid_bbox
   - |bool|
   - If True, use the bounding box information contained in the 
     loaded file. (Default: False)

 * - wrap_mode
   - |string|
   - Controls the behavior of texture evaluations that fall outside of the [0,1]
     range. The following options are currently available:
     - ``repeat``: tile the texture infinitely
     - ``mirror``: mirror the texture along its boundaries
     - ``clamp`` (default): clamp coordinates to the edge of the texture

 * - lambda_min
   - |float|
   - Lower bound of the covered spectral interval

 * - lambda_max
   - |float|
   - Upper bound of the covered spectral interval

 */
enum class SpectrumType { Regular };
enum class FilterType { Trilinear };
enum class WrapMode { Repeat, Mirror, Clamp };

// Forward declaration of specialized GridVolumeSpectral
template <typename Float, typename Spectrum, SpectrumType SpecType>
class GridVolumeSpectralImpl;

/**
 * Interpolated 3D grid texture of spectral values.
 *
 * This plugin loads spectral data from a binary file.
 *
 * Data layout:
 * The data must be ordered so that the following C-style (row-major) indexing
 * operation makes sense after the file has been mapped into memory:
 *     data[((zpos*yres + ypos)*xres + xpos)*channels + chan]}
 *     where (xpos, ypos, zpos, chan) denotes the lookup location.
 *
 * Spatial mesh: values are given at cell centers (classical bitmap mapping)
 * Spectral mesh: values are given at cell nodes (similar to regular mesh
 * layout)
 *
 */
template <typename Float, typename Spectrum>
class GridVolumeSpectral final : public Volume<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Volume, m_world_to_local)
    MTS_IMPORT_TYPES()

    GridVolumeSpectral(const Properties &props) : Base(props), m_props(props) {
        // If used in nonspectral mode, raise
        if constexpr (!is_spectral_v<Spectrum>)
            Throw("This volume data source can only be used with a spectral "
                  "variant!");

        // Spatial interpolation specification parameters
        // NOTE: this is currently a placeholder
        std::string filter_type = props.string("filter_type", "trilinear");
        if (filter_type == "trilinear")
            m_filter_type = FilterType::Trilinear;
        else
            Throw("Invalid filter type \"%s\", must be \"trilinear\"!",
                  filter_type);

        std::string wrap_mode = props.string("wrap_mode", "clamp");
        if (wrap_mode == "repeat")
            m_wrap_mode = WrapMode::Repeat;
        else if (wrap_mode == "mirror")
            m_wrap_mode = WrapMode::Mirror;
        else if (wrap_mode == "clamp")
            m_wrap_mode = WrapMode::Clamp;
        else
            Throw("Invalid wrap mode \"%s\", must be one of: \"repeat\", "
                  "\"mirror\", or \"clamp\"!",
                  wrap_mode);

        // Spectral interpolation parameters
        // NOTE: this is currently a placeholder
        std::string spectrum_type = props.string("spectrum_type", "regular");
        if (spectrum_type == "regular")
            m_spectrum_type = SpectrumType::Regular;
        else
            Throw("Invalid spectrum type \"%s\", must be \"regular\"!",
                  spectrum_type);

        // Load data file into buffer
        auto [metadata, raw_data] =
            read_binary_volume_data<Float>(props.string("filename"));
        m_metadata        = metadata;
        ScalarUInt32 size = hprod(m_metadata.shape);

        m_data = DynamicBuffer<Float>::copy(raw_data.get(),
                                            size * m_metadata.channel_count);

        // Mark values which are only used in the implementation class as
        // queried
        props.mark_queried("use_grid_bbox");
        props.mark_queried("lambda_min");
        props.mark_queried("lambda_max");
    }

    template <SpectrumType SpecType>
    using Impl = GridVolumeSpectralImpl<Float, Spectrum, SpecType>;

    /**
     * Recursively expand into an implementation specialized to the actual
     * loaded grid.
     */
    std::vector<ref<Object>> expand() const override {
        ref<Object> result;
        switch (m_spectrum_type) {
            case SpectrumType::Regular:
                result = (Object *) new Impl<SpectrumType::Regular>(
                    m_props, m_metadata, m_data, m_filter_type, m_wrap_mode);
                break;
            default:
                Throw("Unsupported spectrum type");
        }
        return { result };
    }

    MTS_DECLARE_CLASS()
protected:
    DynamicBuffer<Float> m_data;
    VolumeMetadata m_metadata;
    Properties m_props;
    FilterType m_filter_type;
    WrapMode m_wrap_mode;
    SpectrumType m_spectrum_type;
};

template <typename Float, typename Spectrum, SpectrumType SpecType>
class GridVolumeSpectralImpl final : public Volume<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(Volume, update_bbox, m_world_to_local)
    MTS_IMPORT_TYPES()

    GridVolumeSpectralImpl(const Properties &props, const VolumeMetadata &meta,
                           const DynamicBuffer<Float> &data,
                           FilterType filter_type, WrapMode wrap_mode)
        : Base(props), m_data(data), m_metadata(meta),
          m_inv_resolution_x((int) m_metadata.shape.x()),
          m_inv_resolution_y((int) m_metadata.shape.y()),
          m_inv_resolution_z((int) m_metadata.shape.z()),
          m_filter_type(filter_type), m_wrap_mode(wrap_mode) {

        m_size = hprod(m_metadata.shape);
        if (props.bool_("use_grid_bbox", false)) {
            m_world_to_local = m_metadata.transform * m_world_to_local;
            update_bbox();
        }

        if constexpr (SpecType == SpectrumType::Regular) {
            m_lambda_min   = props.float_("lambda_min");
            m_lambda_max   = props.float_("lambda_max");
            m_dlambda      = m_lambda_max - m_lambda_min;
            m_lambda_scale = m_metadata.channel_count - 1;
        }
    }

    UnpolarizedSpectrum eval(const Interaction3f &it,
                             Mask active) const override {
        ENOKI_MARK_USED(it);
        ENOKI_MARK_USED(active);

        auto result = eval_impl(it, active);
        return result;
    }

    Float eval_1(const Interaction3f &it, Mask active = true) const override {
        ENOKI_MARK_USED(it);
        ENOKI_MARK_USED(active);
        // TODO: check this

        NotImplementedError("eval_1");

        auto result = eval_impl(it, active);
        return hmean(result);
    }

    Vector3f eval_3(const Interaction3f &it,
                    Mask active = true) const override {
        ENOKI_MARK_USED(it);
        ENOKI_MARK_USED(active);

        Throw("eval_3(): The GridVolumeSpectral texture %s was queried for a "
              "3D vector, but this is not supported.",
              to_string());
    }

    MTS_INLINE auto eval_impl(const Interaction3f &it, Mask active) const {
        MTS_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);

        using ResultType = UnpolarizedSpectrum;

        if (none_or<false>(active))
            return zero<ResultType>();

        auto p = m_world_to_local * it.p;
        Wavelength wavelengths_normalized =
            (it.wavelengths - m_lambda_min) / m_dlambda;

        // Spatial and spectral interpolation
        ResultType result = interpolate(p, wavelengths_normalized, active);

        return select(active, result, zero<ResultType>());
    }

    template <typename T> T wrap(const T &value) const {
        if (m_wrap_mode == WrapMode::Clamp) {
            return clamp(value, 0, m_metadata.shape - 1);
        } else {
            T div = T(m_inv_resolution_x(value.x()),
                      m_inv_resolution_y(value.y()),
                      m_inv_resolution_z(value.z())),
              mod = value - div * m_metadata.shape;

            masked(mod, mod < 0) += T(m_metadata.shape);

            if (m_wrap_mode == WrapMode::Mirror)
                mod = select(eq(div & 1, 0) ^ (value < 0), mod,
                             m_metadata.shape - 1 - mod);

            return mod;
        }
    }

    template <typename T> T wrap_wavelengths(const T &value) const {
        // Currently we support only clamping
        return clamp(value, 0, m_metadata.channel_count - 1);
    }

    /**
     * Taking a 3D point in [0, 1)^3, estimates the grid's value at that
     * point using trilinear interpolation.
     *
     * The passed `active` mask must disable lanes that are not within the
     * domain.
     */
    MTS_INLINE auto interpolate(Point3f p, const Wavelength &wavelengths,
                                Mask active) const {
        // If used in nonspectral mode, raise
        if constexpr (!is_spectral_v<Spectrum>) {
            Throw("This volume data source can only be used with a spectral "
                  "variant!");
            // We anyway return something so that the compiler can deduce the
            // returned type
            return zero<UnpolarizedSpectrum>();
        } else {

            using USpectrum     = UnpolarizedSpectrum;
            using USpectrumMask = mask_t<USpectrum>;

            if constexpr (!is_array_v<Mask>)
                active = true;

            using StorageType = DynamicBuffer<Float>;

            const uint32_t nx    = m_metadata.shape.x();
            const uint32_t ny    = m_metadata.shape.y();
            size_t channel_count = m_metadata.channel_count;

            // Lookup neighbouring data points

            /// Spatial component
            //// Scale to bitmap resolution and apply shift
            p = fmadd(p, m_metadata.shape, -.5f);
            //// Integer pixel positions for linear interpolation
            Vector3i p_i = enoki::floor2int<Vector3i>(p);

            /// Spectral component
            //// Scale to range
            auto wavelengths_scaled = wavelengths * m_lambda_scale;
            //// Spectrum index positions for trilinear interpolation
            using USpectrumIndex = int32_array_t<USpectrum>;
            USpectrumIndex wavelengths_i =
                floor2int<USpectrumIndex>(wavelengths_scaled);
            
            // Retrieve data points from storage
            using Int16  = Array<USpectrumIndex, 16>;
            using Int316 = Array<Int16, 3>;
            /// Compute per-axis index values for spatial component
            Int316 index_i_space = wrap(Int316(
                Int16(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1) + p_i.x(),
                Int16(0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1) + p_i.y(),
                Int16(0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1) + p_i.z()
            ));
            /// Compute per-axis index values for spectral component
            Int16 index_i_spectrum = wrap_wavelengths(
                Int16(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1) + wavelengths_i
            );
            /// Compute linear index (in storage) for each spectral component
            Int16 index =
                fmadd(fmadd(fmadd(index_i_space.z(), ny, index_i_space.y()), nx,
                            index_i_space.x()),
                      channel_count, index_i_spectrum);
            
            // Apply the 4-linear interpolation formula
            //// Get values at nodes
            USpectrum d0000 = gather<USpectrum>(m_data, index[0]),
                      d1000 = gather<USpectrum>(m_data, index[1]),
                      d0100 = gather<USpectrum>(m_data, index[2]),
                      d1100 = gather<USpectrum>(m_data, index[3]),
                      d0010 = gather<USpectrum>(m_data, index[4]),
                      d1010 = gather<USpectrum>(m_data, index[5]),
                      d0110 = gather<USpectrum>(m_data, index[6]),
                      d1110 = gather<USpectrum>(m_data, index[7]),
                      d0001 = gather<USpectrum>(m_data, index[8]),
                      d1001 = gather<USpectrum>(m_data, index[9]),
                      d0101 = gather<USpectrum>(m_data, index[10]),
                      d1101 = gather<USpectrum>(m_data, index[11]),
                      d0011 = gather<USpectrum>(m_data, index[12]),
                      d1011 = gather<USpectrum>(m_data, index[13]),
                      d0111 = gather<USpectrum>(m_data, index[14]),
                      d1111 = gather<USpectrum>(m_data, index[15]);
            //// Compute interpolation weights
            Point3f w_space_1 = p - Point3f(p_i), w_space_0 = 1.f - w_space_1;
            USpectrum w_spectrum_1 =
                          wavelengths_scaled - USpectrum(wavelengths_i),
                      w_spectrum_0 = 1.f - w_spectrum_1;
            //// Apply interpolation formula
            USpectrum d000 = fmadd(w_space_0.x(), d0000, w_space_1.x() * d1000),
                      d001 = fmadd(w_space_0.x(), d0001, w_space_1.x() * d1001),
                      d010 = fmadd(w_space_0.x(), d0010, w_space_1.x() * d1010),
                      d011 = fmadd(w_space_0.x(), d0011, w_space_1.x() * d1011),
                      d100 = fmadd(w_space_0.x(), d0100, w_space_1.x() * d1100),
                      d101 = fmadd(w_space_0.x(), d0101, w_space_1.x() * d1101),
                      d110 = fmadd(w_space_0.x(), d0110, w_space_1.x() * d1110),
                      d111 = fmadd(w_space_0.x(), d0111, w_space_1.x() * d1111);
            USpectrum d00  = fmadd(w_space_0.y(), d000, w_space_1.y() * d100),
                      d01  = fmadd(w_space_0.y(), d001, w_space_1.y() * d101),
                      d10  = fmadd(w_space_0.y(), d010, w_space_1.y() * d110),
                      d11  = fmadd(w_space_0.y(), d011, w_space_1.y() * d111);
            USpectrum d0   = fmadd(w_space_0.z(), d00, w_space_1.z() * d10),
                      d1   = fmadd(w_space_0.z(), d01, w_space_1.z() * d11);
            USpectrum result = fmadd(w_spectrum_0, d0, w_spectrum_1 * d1);

            ENOKI_MARK_USED(wavelengths);


            // Deactivate out-of-range wavelengths (probably suboptimal)
            USpectrumMask active_wavelengths =
                wavelengths >= m_lambda_min && wavelengths <= m_lambda_max;
            
            // return result;
            return select(active_wavelengths, result, 0.f);
        }
    }

    ScalarFloat max() const override { return m_metadata.max; }
    ScalarVector3i resolution() const override { return m_metadata.shape; };
    auto data_size() const { return m_data.size(); }

    void traverse(TraversalCallback *callback) override {
        // TODO: implement this
        callback->put_parameter("data", m_data);
        callback->put_parameter("size", m_size);
        Base::traverse(callback);
    }

    void
    parameters_changed(const std::vector<std::string> & /*keys*/) override {

        NotImplementedError("parameters_changed");

        // TODO: implement this
        // auto new_size = data_size();
        // if (m_size != new_size) {
        //     // Only support a special case: resolution doubling along all
        //     axes if (new_size != m_size * 8)
        //         Throw("Unsupported GridVolumeSpectral data size update: %d ->
        //         %d. Expected %d or %d "
        //               "(doubling "
        //               "the resolution).",
        //               m_size, new_size, m_size, m_size * 8);
        //     m_metadata.shape *= 2;
        //     m_size = (ScalarUInt32) new_size;
        // }

        // auto sum = hsum(hsum(detach(m_data)));
        // m_metadata.mean = (double) enoki::slice(sum, 0) / (double) (m_size *
        // 3); if (!m_fixed_max) {
        //     auto maximum = hmax(hmax(m_data));
        //     m_metadata.max = slice(maximum, 0);
        // }
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "GridVolumeSpectral[" << std::endl
            << "  world_to_local = " << m_world_to_local << "," << std::endl
            << "  dimensions = " << m_metadata.shape << "," << std::endl
            << "  mean = " << m_metadata.mean << "," << std::endl
            << "  max = " << m_metadata.max << "," << std::endl
            << "  channels = " << m_metadata.channel_count << std::endl
            << "  lambda_min = " << m_lambda_min << std::endl
            << "  lambda_max = " << m_lambda_max << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
protected:
    DynamicBuffer<Float> m_data;
    VolumeMetadata m_metadata;
    enoki::divisor<int32_t> m_inv_resolution_x;
    enoki::divisor<int32_t> m_inv_resolution_y;
    enoki::divisor<int32_t> m_inv_resolution_z;

    ScalarUInt32 m_size;
    FilterType m_filter_type;
    WrapMode m_wrap_mode;

    ScalarFloat m_lambda_min;
    ScalarFloat m_lambda_max;
    ScalarFloat m_dlambda;
    ScalarFloat m_lambda_scale;
};

MTS_IMPLEMENT_CLASS_VARIANT(GridVolumeSpectral, Volume)
MTS_EXPORT_PLUGIN(GridVolumeSpectral, "GridVolumeSpectral texture")

NAMESPACE_BEGIN(detail)
template <SpectrumType SpecType> constexpr const char *gridvolume_spectral_class_name() {
    if constexpr (SpecType == SpectrumType::Regular)
        return "GridVolumeSpectralImpl_Regular";
    // else if constexpr (SpecType == SpectrumType::Irregular)
    //     return "GridVolumeSpectralImpl_Irregular";
    // else
    //     return "GridVolumeSpectralImpl_Uniform";
}
NAMESPACE_END(detail)

template <typename Float, typename Spectrum, SpectrumType SpecType>
Class *GridVolumeSpectralImpl<Float, Spectrum, SpecType>::m_class = new Class(
    detail::gridvolume_spectral_class_name<SpecType>(), "Volume",
    ::mitsuba::detail::get_variant<Float, Spectrum>(), nullptr, nullptr);

template <typename Float, typename Spectrum, SpectrumType SpecType>
const Class *GridVolumeSpectralImpl<Float, Spectrum, SpecType>::class_() const {
    return m_class;
}

NAMESPACE_END(mitsuba)
