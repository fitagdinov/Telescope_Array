

################ SD ANALYSIS MAIN DIRECTORY ####################

$(if $(wildcard $(SDDIR)),,\
	$(error The SDDIR ($(SDDIR)) variable is not properly set))

############################ DST ###############################

DSTDIR=$(SDDIR)/dst2k-ta
$(if $(wildcard $(DSTDIR)),,\
	$(error DSTDIR ($(DSTDIR)) not found))

# LD flags for compiling with DST libraries
DSTLIBS=-L$(DSTDIR)/lib -ldst2k -lm  -lc -lz -lbz2

################### SD BIN, LIBRARIES AND INCLUDES #############

SDLIBDIR = $(SDDIR)/lib
SDINCDIR = $(SDDIR)/inc
SDBINDIR = $(SDDIR)/bin
RTINCDIR = $(SDDIR)/sdfdrt

# GENERAL SD INCLUDES. EACH PROGRAM WILL ADD ITS OWN INCLUDE FLAGS TO THAT.
INCS = -I$(DSTDIR)/inc -I$(SDINCDIR) -I$(RTINCDIR)

############################## ROOT ############################

$(if $(wildcard $(ROOTSYS)),,\
	$(error CERN Root is not installed on this system))

# for compiling with root
ROOTCFLAGS   := $(shell root-config --cflags)
ROOTLIBS     := $(shell root-config --libs)
ROOTVER      := $(shell root-config --version | sed 's/[\.,\/]//g')
# rootcint flags depending on the version
ROOTCINTFLAGS := -c -p # ROOT versions older than 6.20/00
# ROOT versions later than 6.20/00 don't require these flags
ifeq ($(shell test $(ROOTVER) -ge 62000; echo $$?),0)
ROOTCINTFLAGS :=
endif
# Also include the Minuit and Spectrum analysis libraries which are
# not included by default
ROOTLIBS += -lMinuit -lSpectrum

# For programs that need to be compiled either statically or dynamically
# use alternating root libraries.  Default is dynamic but 
# one can make them static by setting the 'statcompile' environmental
# variable
ROOTLIBS_ALT=$(ROOTLIBS)

############################ OPTIMIZATION ######################

ifeq (x$(OPTOPT),x)
OPTOPT = -O3
endif

############################ DEFINITIONS FLAGS ######################

DEFINITIONFLAGS := -D__DSTDIR__=\"${DSTDIR}\"

############################ COMPILERS ##########################

# default is gcc, ar for making libraries
CPP           = g++
CPPFLAGS      = $(ROOTCFLAGS) $(OPTOPT) -Wall -fPIC $(DEFINITIONFLAGS)
CC	      = gcc
CFLAGS        = $(OPTOPT) -Wall -fPIC  $(DEFINITIONFLAGS)
LD            = g++
LDFLAGS       = $(OPTOPT)
AR            = ar

# intel compilers
ifneq (x$(usingicc),x)
CPP           = icpc
CPPFLAGS      = $(ROOTCFLAGS) $(OPTOPT) -ipo -no-prec-div -xHost -fpic  $(DEFINITIONFLAGS)
CC	      = icc
CFLAGS        = $(OPTOPT) -ipo -no-prec-div -xHost -fpic  $(DEFINITIONFLAGS)
LD            = icpc
LDFLAGS       = $(OPTOPT)
AR            = xiar
endif

############################# FOR A STATIC COMPILE ######################

ROOTLIBS_STATIC = -L$(shell root-config --libdir) -lRoot -lpcre -lncurses 
ROOTLIBS_STATIC += -lpthread -lpthread_nonshared


# in case of a static compile, should
# use static root libraries in those programs
# that support the static compile with root 
# ( programs that use ROOTLIBS_ALT rather than just
# ROOTLIBS )
ifneq (x$(statcompile),x)
ROOTLIBS_ALT=$(ROOTLIBS_STATIC)

# additional linking option to use when
# using the intel compiler(s) and compiling
# everything static
ifneq (x$(usingicc),x)
LDFLAGS += -fast
endif


endif

########################### USEFUL SUFFIX RULES #########################

%.o : %.cpp ; \
$(CPP) $(CPPFLAGS) $(INCS) -o $@ $< -c

%.o : %.cxx ; \
$(CPP) $(CPPFLAGS) $(INCS) -o $@ $< -c

%.o : %.c ; \
$(CC) $(CFLAGS) $(INCS) -o $@ $< -c


########################### OTHER ########################################

.PHONY: all clean cleanall
