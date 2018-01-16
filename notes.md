## Next steps

- APOGEE and KIC matching, using RA and Declination, load 

PS and APOGEE matching first for the dataset
Draw line in delta mu and PS space and use this as ground truth to compute
https://arxiv.org/pdf/1602.04940.pdf

Set up meeting with Henry to get the data

Hydrogen shell burning in RGB first then He flash

Look at:
http://astronn.readthedocs.io/en/latest/tools_apogee.html

`gap_delete` - deletes gaps in the spectra


## Questions

What kind of data is easily available? (high level overview of the problem)
Why is spectra more widely available for stars than seismic?

Is the current way of prediction of RC mainly done through asteroseismology? It seems like it is and involves the mentioned parameters, \nabla{}v and others, what is the closed form and what is the contaminiation rate? (Hawkins mentions ~9% in Bovy). 

Why is it difficult and what can I assume is ground truth with PS is not accurate? (is it?). 

Point of detecting secondary RC? Is there a point to distinguish them since they are not standard candles?


## Points

Decreasing resolution of spectra and its impact on prediction performance
Few-shot learning to avoid overfitting
Is the problem actually prediction of PS or binary classification of RC? It seems that PS is used to predict RC status. 


## Useful links

APOGEE and KIC star catalog explanation:
https://archive.stsci.edu/kepler/red_giant_release.html
tl;dr, red giant data provided by Kepler (period and other important stellar parameters)
APOGEE provides high quality spectra

http://www.skyandtelescope.com/astronomy-resources/what-are-celestial-coordinates/

# Notes on terminology

> Abundance: The abundance of the chemical elements is a measure of the occurrence of the chemical elements relative to all other elements in a given environment.

> Signal: A signal as referred to in communication systems, signal processing, and electrical engineering is a function that "conveys information about the behavior or attributes of some phenomenon"

> Sun-like star brightens during hydrogen fusion as the helium ash core builds up, creating a denser core that fuses hydrogen at a faster rate: star brightens, eventually core contracts due to depletion of fuel and pressure rises that fusion is once agian possible and the star exits the Main Sequence as it expands: star cools and dims: Red giant branch

RGB-stars have inert helium core and are still fusing hydrogen in their cores and the helium ash core is continuously added to. Once hydrogen is depleted, core contracts before helium fusion is possible through triple-alpha process: 3 He -> 1 C: helium flash. Occurs suddenly; larger stars will have thicker cores that can begin helium fusion sooner.

This asymptotic approach makes stars dim but remain slightly hotter than the same red giant branch stars that are still moving towards He flash. He flash means the stars cores absorbs most of the energy of the flash to create a buried white dwarf and a constant stable Helium fusion: Subgiant star. Helium fused into carbon faster than H -> He since it is heavier, but no carbon flash since core is not heavy enough to begin fusion of it. 



# An Application of Deep Neural Networks in Analysis of Stellar Spectra
https://arxiv.org/pdf/1709.09182.pdf

Standard CNN with 2 dense layers at the end for regression prediction on:
	1. Effective surface temperature
	2. Surface gravity log(g)
	3. Metallicity [Fe/H]

	-- Question: What importance do these stellar parameters hold?

They use ADAM 

# The Cannon 2 - Data Driven Model for Stellar Spectra for detailed Chemical abundance analyses

	-- Question: What is the term abundance and how is this relevant?
	-- Why is this relevant? And how can it be measured from the interior of a star anyways?

Large dimension prediction space - basically R^10.


# Apogee Data Red-Clump stars

-- first parameter is location of sky
-- name of star

-- astropy


# Related papers
https://arxiv.org/pdf/1712.02405.pdf
http://www.sdss.org/dr14/data_access/bulk/
https://www.aanda.org/articles/aa/pdf/2016/04/aa27259-15.pdf
https://www.aanda.org/articles/aa/pdf/2016/04/aa27259-15.pdf
https://ui.adsabs.harvard.edu/?#abs/2016A%26A...588A..87V
http://docs.astropy.org/en/stable/io/unified.html#ascii-formats