###################### CHECKS ##################################

############### dst2k-ta integrity (below) ###############

# presence of $DSTDIR/inc
$(if $(wildcard $(DSTDIR)/inc),,$(error $(DSTDIR)/inc not found))

# presence of $DSTDIR/lib
$(if $(wildcard $(DSTDIR)/lib),,$(error $(DSTDIR)/lib not found))

# presence of $DSTDIR/inc/event.h
have_event_h=$(shell test -f $(DSTDIR)/inc/event.h && echo "have_event_h")
$(if $(have_event_h),,$(error $$DSTDIR/inc/event.h not found))

# presence of libdst2k.a not checked here because it will be created
# at the compilation stage

############### dst2k-ta integrity (above) ################

############### CERN Root installation (below) ############

# ROOTSYS environmental variable
$(if $(wildcard $(ROOTSYS)),,$(error CERN Root not properly installed. \
        Visit http://www.telescopearray.org/tawiki/index.php/Root for more information))

# root-config utility
have_root=$(shell root-config --version | head -1 | awk '{print $$1}')
$(if $(have_root),,$(error CERN Root installation has not been completed. You need to \
        source $$ROOTSYS/bin/thisroot.sh (for bash) or source $$ROOTSYS/bin/thisroot.csh (for C-shell). \
        For more information, visit http://www.telescopearray.org/tawiki/index.php/Root))

# rootcint utility
have_rootcint=$(shell rootcint -h 2>&1 | grep -i linkdef | head -1 | awk '{print $$1}')
$(if $(have_rootcint),,$(error command 'rootcint' is either missing or not working. Re-install CERN Root. \
        Visit http://www.telescopearray.org/tawiki/index.php/Root for more information))

############# CERN Root installation (above) ##############

# Making sure that all libraries and headers are found (below) #

# for compiling without forcing the static compile mode
have_incs_libs=$(shell $(CPP) $(CPPFLAGS) $(INCS) $(SDDIR)/sdanalysis_checks.cpp \
	-o $(SDBINDIR)/sdanalysis_checks.run -lz -lbz2 $(ROOTLIBS) >/dev/null; \
	test $$? -eq 0 && echo "have_everything")
$(if $(have_incs_libs),,$(error SOMETHING IS MISSING))

# for compiling with forcing the static mode (becomes relevant if one
# use statcompile=1 option, then ROOTLIBS_ALT and ROOTLIBS are not the
# same thing.
have_incs_libs=$(shell $(CPP) $(CPPFLAGS) $(INCS) $(SDDIR)/sdanalysis_checks.cpp \
	-o $(SDBINDIR)/sdanalysis_checks.run -lz -lbz2 $(ROOTLIBS_ALT) >/dev/null; \
	test $$? -eq 0 && echo "have_everything")
$(if $(have_incs_libs),,$(error SOMETHING IS MISSING))

# Making sure that all libraries and headers are found (above) #

