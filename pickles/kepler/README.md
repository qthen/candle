# Seismic global parameters of 6111 KIC pickles

Pickles for KIC data from `Vrad 2016` and their related spectra. This includes the follow dict struct:

	{ 
		KIC     : [Int]
		RA      : [Float]
		DEC     : [Float]
		spectra : [[Float]]
		Dnu     : [Float]
		PS      : [Float]
		T_eff   : [Float]
		logg    : [Float]
	}

Formed by populating stars from `Seismic global parameters of 6111 KIC` with their RA and DEC, then xmatch them with APOGEE(dr=14) spectra, then deleting the pre-defined gaps from DR=14

## stars1.pickle
Contains all stars that can be xmatch from `Vrad 2016` with APOGEE(dr=14)

## stars2.pickle
Contains all stars subject to the same constraints as Hawkins et. al. paper, that is that:

1. APOGEE_STARFLAG is 0
2. ASPCAP flag is 0
3. Uncertainity of PS (DPi1) is less than 10s
