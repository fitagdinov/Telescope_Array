
INCS += -I$(RUFLDF)/inc
rufldf_libs = -L$(SDLIBDIR) -lsden -lsduti
rufldf_libs += $(ROOTLIBS_ALT) $(DSTLIBS) 

rufldf_cpplist = rufldf.cpp rufldf_util.cpp rufldfAnalysis.cpp
rufldf_cpplist += p2geomfitter.cpp p2ldffitter.cpp p2gldffitter.cpp 
rufldf_srcs = $(addprefix $(RUFLDF)/src/, $(rufldf_cpplist))
rufldf_objs = ${rufldf_srcs:.cpp=.o}

SDBINS += $(SDBINDIR)/rufldf.run

.PHONY: rufldf
rufldf: $(SDBINDIR)/rufldf.run
all:: rufldf

$(SDBINDIR)/rufldf.run:$(rufldf_objs) ; \
$(LD) $(LDFLAGS) $^ $(rufldf_libs) -o $@ ; \

clean:: ; \
rm -f $(rufldf_objs) $(RUFLDF)/*~  $(RUFLDF)/src/*~; \
