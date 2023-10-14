# to compile libraries common to all SD analysis programs.
SHELL    =/bin/bash

# SD ANALYSIS COMPILATION SETTINGS
SDDIR=$(shell pwd)
include $(SDDIR)/makefileset.mk

# DO THE CHECKS
include $(SDDIR)/sdanalysis_checks.mk

# making sure that the default make target is all (included mk files are
# adding their programs/libraries to the "all" target)
all::

# list of SD analysis libraries and binaries initialized here.  
# each .mk file if it builds any binaries or libraries will add to the lists
# corresponding entries
SDBINS=
SDLIBS=

# all SD shared libraries that will be built initlized
# makefile includes will add shared libraries to this variable
SDSOLIBS=

# some SD shared libraries need to be lodable from Root by
# themselves, so they should link in all SD libraries
# stands for "SD stand alone shared libraries"
# make file includes will add shared libraries to this variable
SDSASOLIBS=

# raw SD data parser
RUSDPASS0=$(SDDIR)/rusdpass0
include $(RUSDPASS0)/rusdpass0.mk

# signal analysis + pattern recognition + simple geometry
RUFPTN=$(SDDIR)/rufptn
include $(RUFPTN)/rufptn.mk

# geometry and LDF
RUFLDF=$(SDDIR)/rufldf
include $(RUFLDF)/rufldf.mk

# histogramming
RUSDHIST=$(SDDIR)/rusdhist
include $(RUSDHIST)/rusdhist.mk

# printing the SD analysis ASCII files
SDASCII=$(SDDIR)/sdascii
include $(SDASCII)/sdascii.mk

# trigger backup programs
TRIG=$(SDDIR)/trig
include $(TRIG)/trig.mk

# simple iterating program (iterates over DST files)
SDITERATOR=$(SDDIR)/sditerator
include $(SDITERATOR)/sditerator.mk

# making (mostly SD - oriented) root trees
DST2RT_SD=$(SDDIR)/dst2rt_sd
include $(DST2RT_SD)/dst2rt_sd.mk

# make root trees only with the thrown SD MC variables
PREPMC=$(SDDIR)/prepmc
include $(PREPMC)/prepmc.mk

# miscellaneous small programs that should be included in the stable
# release
MISC=$(SDDIR)/misc
include $(MISC)/misc.mk

# atmospheric calibration
# atmospheric correction
# fitting of atmospheric parameters for simulations
SDATM=$(SDDIR)/sdatm
include $(SDATM)/sdatm.mk

# SD Monte Carlo
SDMC=$(SDDIR)/sdmc
include $(SDMC)/sdmc.mk

# utilities to convert SD MC to tasdcalibev format
MC2TASDCALIBEV=$(SDDIR)/mc2tasdcalibev
include $(MC2TASDCALIBEV)/mc2tasdcalibev.mk

# interactive geometry fitting so-file
SDGEOMFITTER = $(SDDIR)/sdgeomfitter
include $(SDGEOMFITTER)/sdgeomfitter.mk

# interactive LDF fitting
LDFFITTER = $(SDDIR)/ldffitter
include $(LDFFITTER)/ldffitter.mk

# interactive Geometry/LDF fitting
GLDFFITTER = $(SDDIR)/gldffitter
include $(GLDFFITTER)/gldffitter.mk

# SD event display
PASS1PLOT = $(SDDIR)/pass1plot
include $(PASS1PLOT)/pass1plot.mk
# pass1 plot is the event display shared
# library that has to be loadable from Root
# (and it loads all other shared-library moduli)
# so pass1plot has to link in all SD libraries
# that are needed by any other shared library modulus
pass1plot: $(SDLIBS)

# library for the energy estimation table(s)
SDENERGY=$(SDDIR)/sdenergy
include $(SDENERGY)/sdenergy.mk

# library with SD utilities
SDUTI=$(SDDIR)/sduti
include $(SDUTI)/sduti.mk

# library with (mostly SD-oriented) root-tree classes
SDFDRT=$(SDDIR)/sdfdrt
include $(SDFDRT)/sdfdrt.mk

# building DST libraries using SD analysis options and settings
include $(SDDIR)/sdanalysis_dst2k-ta.mk

# SD analysis scripts
SDSCRIPTS=$(SDDIR)/sdscripts
include $(SDSCRIPTS)/sdscripts.mk

libs: $(SDLIBS) $(SDSOLIBS)

# This target 'cleanall' will clean every directory and 
# also delete the binary files and libraries
cleanall: clean ; \
rm -f $(SDBINDIR)/sdanalysis_checks.run $(SDBINS) $(SDLIBS) $(SDSOLIBS) $(SDLIBDIR)/*.pcm

# Dependencies of binaries on SD libraries (not including
# here root tree and/or shared libraries which sould be done
# separately because not all programs depend on them)
$(SDBINS): $(SDLIBS)

#  SD stand-alone shared libraries depend on all SD static libraries
$(SDSASOLIBS): $(SDLIBS)

# Display the post-compile message if successful
all:: build_successful
build_successful: $(SDBINS) $(SDSOLIBS) $(SDSASOLIBS) $(sdscripts)
	@echo ""
	@echo "====================="
	@echo "  Compile complete!"
	@echo "====================="
	@echo "Now run (in Bash):"
	@echo "source sdanalysis_env.sh"
	@echo "to finish the setup.  Add line"
	@echo "source $$SDDIR/sdanalysis_env.sh"
	@echo "to your \$$HOME/.bashrc file"
	@echo ""
