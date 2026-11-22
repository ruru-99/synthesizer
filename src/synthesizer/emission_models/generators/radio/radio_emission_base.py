"""A submodule defining the base class for radio power-law generators.

The emission is a power law in frequency,

    L_nu = S * norm * (nu / nu_0) ** index,

where the scaling S depends on the mode:

    * Mass mode (no scaler/intrinsic/attenuated given): S is the gas mass
      from the emitter, and norm has units erg s^-1 Hz^-1 Msun^-1.
    * Scaled mode (scaler given): S is the bolometric luminosity of the
      scaler emission, and norm = L_nu(nu_0) / L_bol has units Hz^-1.
    * Energy-balance mode (intrinsic and attenuated given): S is the
      bolometric luminosity absorbed by dust (intrinsic - attenuated), and
      norm again has units Hz^-1.

A power law with index >= -1 (or <= -1) has no finite bolometric integral
over all frequencies, so the normalisation is always defined at nu_0 rather
than as a fraction of the bolometric luminosity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from unyt import Hz, Msun, Unit, angstrom, c, erg, s, unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.emission_models.base_model import EmissionModel
from synthesizer.emission_models.generators.dust.dust_emission_base import (
    DustEmission,
)
from synthesizer.emissions import LineCollection, Sed
from synthesizer.units import Units, accepts
from synthesizer.utils.operation_timers import timed
from synthesizer.utils.precision import verify_out_precision

if TYPE_CHECKING:
    from synthesizer.components.component import Component


def _as_quantity(value, units):
    """Attach units to a bare number, or convert a unyt object to units."""
    if hasattr(value, "units"):
        return value.to(units)
    return unyt_quantity(value, units)


class RadioEmission(DustEmission):
    """Radio power-law generator, scaled by gas mass by default.

    Inherits the scaler / energy-balance machinery from DustEmission, so the
    mode is chosen by which emission models are passed in. Use RadioScaled
    and RadioEnergyBalance for the two luminosity-scaled modes.
    """

    def __init__(
        self,
        normalisation,
        index: float = -1.0,
        nu_0=100.0 * Hz,
        intrinsic: Optional[EmissionModel] = None,
        attenuated: Optional[EmissionModel] = None,
        scaler: Optional[EmissionModel] = None,
    ) -> None:
        """Initialise the radio generator.

        Args:
            normalisation (float/unyt_quantity):
                Mass mode: erg s^-1 Hz^-1 Msun^-1 (bare floats assumed to be
                in these units). Scaled / energy-balance modes: Hz^-1, i.e.
                L_nu(nu_0) per unit scaling luminosity.
            index (float):
                The power-law index.
            nu_0 (float/unyt_quantity):
                The characteristic frequency (bare floats assumed Hz).
            intrinsic (EmissionModel):
                Intrinsic emission for energy balance.
            attenuated (EmissionModel):
                Attenuated emission for energy balance.
            scaler (EmissionModel):
                Emission whose bolometric luminosity scales the radio.
        """
        mass_mode = scaler is None and (intrinsic is None or attenuated is None)

        DustEmission.__init__(
            self,
            intrinsic,
            attenuated,
            scaler,
            False,  # no CMB heating for radio
            required_params=("masses",) if mass_mode else (),
        )

        self.index = index
        self.nu_0 = _as_quantity(nu_0, Hz)
        self.normalisation = _as_quantity(
            normalisation,
            Hz**-1 if not mass_mode else erg / s / Hz / Msun,
        )

    def __repr__(self):
        """Return a string representation of the generator."""
        mode = (
            "energy_balance"
            if self.is_energy_balance
            else "scaled"
            if self.is_scaled
            else "mass"
        )
        return (
            f"{self.__class__.__name__}(mode={mode}, "
            f"norm={self.normalisation}, index={self.index}, "
            f"nu_0={self.nu_0})"
        )

    def _shape(self, nu: unyt_array) -> np.ndarray:
        """Return (nu / nu_0) ** index."""
        return (nu / self.nu_0).to("dimensionless").value ** self.index

    def get_scaling(self, emitter, model, emissions):
        """Return the scaling in luminosity / Hz, including the norm.

        With no emitter (standalone get_spectra) this is 1.0, so the pure
        shape (nu / nu_0) ** index is returned.
        """
        if emitter is None:
            return 1.0

        out_units = Unit(Units().luminosity) / Hz

        # Luminosity-scaled modes: scaler or energy balance
        if self.is_energy_balance or self.is_scaled:
            lscale = DustEmission.get_scaling(self, emitter, model, emissions)
            return (lscale * self.normalisation).to(out_units)

        # Mass mode
        masses = self._extract_params(model, emitter)["masses"]
        masses = (
            masses.to(Msun)
            if hasattr(masses, "units")
            else unyt_array(masses, Msun)
        )
        if model is None or not model.per_particle:
            masses = masses.sum()
        return (masses * self.normalisation).to(out_units)

    @accepts(lams=angstrom)
    @timed("RadioEmission._generate_spectra")
    @verify_out_precision()
    def _generate_spectra(
        self,
        lams: unyt_array,
        emitter: Component,
        model: EmissionModel,
        emissions: dict,
        redshift: float = 0,
        out_dtype=None,
    ) -> Sed:
        """Generate the radio emission spectra.

        Args:
            lams (unyt_array):
                The wavelength grid (must reach radio wavelengths).
            emitter (Stars/Gas/BlackHole/Galaxy):
                The object emitting the emission.
            model (EmissionModel):
                The emission model generating the emission.
            emissions (dict):
                Dictionary containing all emissions generated so far.
            redshift (float):
                Unused, kept for interface compatibility.
            out_dtype (np.dtype):
                The output precision.

        Returns:
            Sed:
                The generated radio SED.
        """
        nu = (c / lams).to(Hz)
        shape = self._shape(nu)

        sed = Sed(lam=lams, lnu=shape * erg / s / Hz)

        scaling = self.get_scaling(emitter, model, emissions)
        sed._lnu = self.get_scaled_emission(
            shape, scaling, emitter, model, out_dtype
        )
        return sed

    @accepts(line_lams=angstrom)
    @verify_out_precision()
    def _generate_lines(
        self,
        line_ids,
        line_lams,
        emitter,
        model,
        emissions,
        spectra,
        redshift=0,
        out_dtype=None,
    ) -> LineCollection:
        """Generate the radio continuum at the line wavelengths (no lines)."""
        if model.label not in spectra:
            raise exceptions.MissingSpectraType(
                f"Cannot generate lines for {model.label} as spectra are "
                "missing. Please generate spectra first or remove the "
                "generation of lines."
            )

        nu = (c / line_lams).to(Hz)
        shape = self._shape(nu)

        scaling = self.get_scaling(emitter, model, spectra)
        lnu = self.get_scaled_emission(
            shape, scaling, emitter, model, out_dtype
        )

        return LineCollection(
            line_ids,
            line_lams,
            lum=np.zeros(lnu.shape, dtype=lnu.dtype) * erg / s,
            cont=lnu * erg / s / Hz,
        )

    @accepts(lams=angstrom)
    def get_spectra(self, lams: unyt_array, redshift: float = 0) -> Sed:
        """Return the unscaled shape, (nu / nu_0) ** index, on lams."""
        return self._generate_spectra(lams, None, None, {}, redshift)
