
INCS += -I$(RUFPTN)/inc
rufptn_libs = -L$(SDLIBDIR) -lsduti $(ROOTLIBS_ALT) $(DSTLIBS) 
rufptn_cpplist = rufptn.cpp rufptn_util.cpp rufptnAnalysis.cpp p1geomfitter.cpp
rufptn_srcs = $(addprefix $(RUFPTN)/src/, $(rufptn_cpplist))
rufptn_objs = ${rufptn_srcs:.cpp=.o}

SDBINS += $(SDBINDIR)/rufptn.run
.PHONY: rufptn
rufptn: $(SDBINDIR)/rufptn.run
all:: rufptn

$(SDBINDIR)/rufptn.run:$(rufptn_objs) ; \
$(LD) $(LDFLAGS) $^ $(rufptn_libs) -o $@ ; \


clean:: ; \
rm -f $(RUFPTN)/*~ $(rufptn_objs) $(RUFPTN)/src/*~ ; \
