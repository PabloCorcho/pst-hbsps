import numpy as np
import matplotlib.pyplot as plt

import specBasics
from pst import SSP
from pst.utils import flux_conserving_interpolation

class Reader(object):
    def __init__(self, ini_file):
        self.ini_file = ini_file
        self.ini = self.read_ini_file(self.ini_file)
        self.load_chain()
        self.load_observation()
        self.load_ssp()

    def load_chain(self):
        path = self.ini['output']['filename']
        if ".txt" not in path:
            path += ".txt"
        self.chain = self.read_chain_file(path)

    def load_observation(self):
        print("Loading observation input data")
        obs_wl, obs_f, obs_e = np.loadtxt(self.ini['HBSPS_SFH']['inputSpectrum'],
                                          unpack=True)
        obs_wl /= (1 + self.ini['HBSPS_SFH']['redshift'])
        obs_cov = obs_e**2
        # Apply the same pre-processing to the spectra as during the fit
        flux, ln_wave, _ = specBasics.log_rebin(
            obs_wl, obs_f, velscale=self.ini['HBSPS_SFH']['velscale'])
        cov, _, _ = specBasics.log_rebin(
            obs_wl, obs_cov, velscale=self.ini['HBSPS_SFH']['velscale'])
        wavelength = np.exp(ln_wave)

        norm_flux = np.nanmedian(
            flux[(wavelength >= self.ini['HBSPS_SFH']['wlNormRange'][0]) & (
                   wavelength <= self.ini['HBSPS_SFH']['wlNormRange'][1])])
        flux /= norm_flux
        cov /= norm_flux**2
        self.observation = {'wavelength': obs_wl, 'flux': obs_f, 'error': obs_e,
                            'wavelength_fit': wavelength, 'flux_fit': flux, 
                            'cov_fit': cov, 'norm_flux_fit': norm_flux}

    def load_ssp(self):
        print("Loading SSP model: ", self.ini['HBSPS_SFH']['SSPModel'])
        ssp_model = getattr(SSP, self.ini['HBSPS_SFH']['SSPModel'])
        if 'SSPDir' in self.ini['HBSPS_SFH']:
            ssp_dir = self.ini['HBSPS_SFH']['SSPDir']
            if ssp_dir == 'None':
                ssp_dir = None
        else:
            ssp_dir = None
        if 'SSPModelArgs' in self.ini['HBSPS_SFH']:
            ssp_args = self.ini['HBSPS_SFH']['SSPModelArgs']
            ssp_args = ssp_args.split(",")
        else:
            ssp_args = []
        self.ssp = ssp_model(*ssp_args, path=ssp_dir)
        self.ssp.regrid(
            self.ini['HBSPS_SFH']['ageRange'],
            self.ini['HBSPS_SFH']['metRange'])
        self.n_ssp = self.ssp.L_lambda[:, :, 0].size
        print("Number of SSP models used: ", self.n_ssp)
        self.ssp_mlr = self.ssp.get_mass_lum_ratio(self.ini['HBSPS_SFH']['wlNormRange'])
        self.ssp.L_lambda *= self.ssp_mlr[:, :, np.newaxis]

        print("Log-binning SSP spectra to velocity scale: ",
              self.ini['HBSPS_SFH']['velscale'] / self.ini['HBSPS_SFH']['oversampling'], " km/s")
        dlnlam = self.ini['HBSPS_SFH']['velscale'] / specBasics.constants.c.to('km/s').value
        extra_offset_pixel = 300 / self.ini['HBSPS_SFH']['velscale']
        dlnlam /= self.ini['HBSPS_SFH']['oversampling']
        lnlam_bin_edges = np.arange(
		np.log(self.observation['wavelength_fit'])[0]
        - dlnlam * extra_offset_pixel * self.ini['HBSPS_SFH']['oversampling']
        - 0.5 * dlnlam,
        np.log(self.observation['wavelength_fit'])[-1]
        + dlnlam * (1 + extra_offset_pixel) * self.ini['HBSPS_SFH']['oversampling']
        + 0.5 * dlnlam,
        dlnlam)
        self.ssp.interpolate_sed(np.exp(lnlam_bin_edges))
        print("Final SSP model shape: ", self.ssp.L_lambda.shape)

    def get_chain_percentiles(self, pct=[.5, .16, .50, .84, .95]):
        parameters = [par for par in self.chain.keys() if "parameters" in par]
        pct_resutls = {'percentiles': np.array(pct)}
        for par in parameters:
            sort_pos = np.argsort(self.chain[par])
            cum_distrib = np.cumsum(self.chain['weight'][sort_pos])
            cum_distrib /= cum_distrib[-1]
            pct_resutls[par] = np.interp(
                pct, cum_distrib, self.chain[par][sort_pos])
        return pct_resutls

    def compute_solution_from_pct(self, pct_results, pct=0.5):
        print("Computing solution from percentiles")
        column = np.argmin(np.abs(pct_results['percentiles'] - pct))
        los_vel = pct_results['parameters--los_vel'][column]
        redshift = np.exp(
        los_vel / specBasics.constants.c.to('km/s').value) - 1 
        los_sigma = pct_results['parameters--sigma'][column]

        ssp_weights = np.zeros(self.n_ssp)
        for i in range(1, self.n_ssp + 1):
            ssp_weights[i - 1] = 10**pct_results[f'parameters--ssp{i}'][column]
        ssp_weights /= ssp_weights.sum()
        syn_spectra = np.sum(
            ssp_weights.reshape(self.ssp.L_lambda.shape[:-1])[:, :, np.newaxis]
            * self.ssp.L_lambda, axis=(0, 1))
        sigma_pix = los_sigma / (
            self.ini['HBSPS_SFH']['velscale'][0]
            / self.ini['HBSPS_SFH']['oversampling'][0])
        syn_spectra = specBasics.smoothSpectrumFast(
        syn_spectra, sigma_pix)
        syn_spectra = flux_conserving_interpolation(
            self.observation['wavelength_fit'],
            self.ssp.wavelength * (1 + redshift), syn_spectra)
        plt.figure()
        plt.subplot(111)
        plt.fill_between(
            self.observation['wavelength_fit'],
            self.observation['flux_fit'] - self.observation['cov_fit']**0.5,
            self.observation['flux_fit'] + self.observation['cov_fit']**0.5,
            color='k', alpha=0.3)
        plt.plot(self.observation['wavelength_fit'],
                 self.observation['flux_fit'], color='r', label='Obs')
        plt.plot(self.observation['wavelength_fit'],
                 syn_spectra, color='lime', label='Synth')
        plt.annotate(f"v={los_vel:.1f} (km/s)" + "\n" + r"$\sigma=$"
                     + f"{los_sigma:.1f} (km/s)",
                     xy=(0.05, 0.95), xycoords='axes fraction', va='top')
        plt.plot(self.observation['wavelength_fit'],
                 self.observation['flux_fit'] - syn_spectra, color='orange',
                 label='Residual')
        plt.legend()
        plt.show()

    @staticmethod
    def read_ini_file(path):
        print("Reading ini file: ", path)
        ini_info = {}
        with open(path, "r") as f:
            for line in f.readlines():
                line = line.replace("\n", "")
                if len(line) == 0:
                    continue
                if line[0] == '[':
                    module = line.strip("[]")
                    ini_info[module] = {}
                else:
                    components = line.split("=")
                    name = components[0].strip(" ")
                    str_value = components[1].replace(" ", "")
                    str_value = str_value.replace(".", "")
                    str_value = str_value.replace("e", "")
                    str_value = str_value.replace("-", "")
                    if not str_value.isnumeric():
                        ini_info[module][name] = components[1].strip(" ")
                    else:
                        numbers = [n for n in components[1].split(" ") if len(n) > 0]
                        if ("." in components[1]) or ("e" in components[1]):
                            # Float number
                            ini_info[module][name] = np.array(
                                numbers, dtype=float)
                        else:
                            # int number
                            ini_info[module][name] = np.array(
                                numbers, dtype=int)
        return ini_info
    
    @staticmethod
    def read_chain_file(path):
        print("Reading chain results: ", path)
        with open(path, "r") as f:
            header = f.readline().strip("#")
            columns = header.replace("\n", "").split("\t")
        matrix = np.loadtxt(path)
        results = {}
        ssp_weights = np.zeros(matrix.shape[0])
        last_ssp = 0
        for ith, par in enumerate(columns):
            results[par] = matrix[:, ith]
            if 'ssp' in par:
                last_ssp +=1
                ssp_weights += 10**matrix[:, ith]
        print(f"Adding extra SSP {last_ssp + 1}")
        results[f'parameters--ssp{last_ssp + 1}'] = np.log10(
            np.clip(1 - ssp_weights, a_min=1e-4, a_max=None))
        return results