from hbsps.pipeline_modules.base_module import BaseModule
import numpy as np

from cosmosis.datablock import names as section_names
from cosmosis.datablock import SectionOptions
from hbsps import kinematics

class FullSpectralFitModule(BaseModule):
    name = "FullSpectralFit"
    def __init__(self, options):
        """Set-up the COSMOSIS sampler.
            Args:
                options: options from startup file (i.e. .ini file)
            Returns:
                config: parameters or objects that are passed to 
                    the sampler.
                    
        """
        options = self.parse_options(options)
        # Pipeline values file
        self.config = {}
        self.prepare_observed_spectra(options)
        self.prepare_ssp_model(options)
        self.prepare_sfh_model(options)
        self.prepare_extinction_law(options)

    def make_observable(self, block):
        """Create the spectra model from the input parameters"""
        # Stellar population synthesis
        sfh_model = self.config['sfh_model']
        flux_model = sfh_model.model.compute_SED(self.config['ssp_model'],
                                                 t_obs=sfh_model.today,
                                                 allow_negative=False).value

        # Kinematics
        velscale = self.config["velscale"]
        oversampling = self.config["oversampling"]
        # Kinematics
        sigma_pixel = block["parameters", "los_sigma"] / (velscale / oversampling)
        veloffset_pixel = block["parameters", "los_vel"] / (velscale / oversampling)

        # Build the kernel. TOO SLOW? Initialise only once?
        kernel = kinematics.get_losvd_kernel(
            kernel_model = kinematics.GaussHermite(
            4,
            mean=veloffset_pixel,
            stddev=sigma_pixel,
            h3=block["parameters", "los_h3"],
            h4=block["parameters", "los_h4"]),
            x_size=8 * int(np.round(sigma_pixel)) + 1
        )
        # Perform the convolution
        flux_model = kinematics.convolve_spectra_with_kernel(
            flux_model, kernel)
        # Track those pixels at the edges
        mask = flux_model > 0
        mask[: int(5 * sigma_pixel)] = False
        mask[-int(5 * sigma_pixel) :] = False
        # Sample to observed resolution

        extra_pixels = self.config["extra_pixels"]
        flux_model = (
        flux_model[extra_pixels * oversampling : -(1 + extra_pixels * oversampling)]
        .reshape((self.config['flux'].size, oversampling))
        .mean(axis=1))
        mask = (
        mask[extra_pixels * oversampling : -(1 + extra_pixels * oversampling)]
        .reshape((self.config['flux'].size, oversampling))
        .sum(axis=1))
        # Enforce that all combined pixels are good
        mask = mask == oversampling

        # Apply dust extinction
        dust_model = self.config["extinction_law"]
        flux_model = dust_model.apply_extinction(
            self.config['wavelength'], flux_model,
            a_v=block["parameters", "av"]).value

        weights = self.config["weights"] * mask
        normalization = np.sum(
            self.config['flux'] / flux_model * self.config["weights"]
            ) / np.sum(self.config["weights"])
        block['parameters', 'normalization'] = normalization
        return flux_model * normalization, weights


    def execute(self, block):
        """Function executed by sampler
        This is the function that is executed many times by the sampler. The
        likelihood resulting from this function is the evidence on the basis
        of which the parameter space is sampled.
        """
        # Obtain parameters from setup
        cov = self.config['cov']
        flux_model, weights = self.make_observable(block)
        weights_norm = np.nansum(weights)
        # Calculate likelihood-value of the fit
        like = self.log_like(self.config["flux"] * weights,
                             flux_model * weights, cov * weights_norm)
        # Final posterior for sampling
        block[section_names.likelihoods, f"{self.name}_like"] = like
        return 0

    def cleanup(self):
        pass


def setup(options):
        options = SectionOptions(options)
        mod = FullSpectralFitModule(options)
        return mod

def execute(block, mod):
    mod.execute(block)
    return 0

def cleanup(mod):
    mod.cleanup()