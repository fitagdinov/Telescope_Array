
INCS += -I$(RUSDHIST)/inc
rusdhist_libs = -L$(SDLIBDIR) -lsduti $(ROOTLIBS_ALT) $(DSTLIBS) 

rusdhist_cpplist = rusdhist.cpp rusdhist_util.cpp rusdhist_class.cpp 
rusdhist_cpplist += sdenergy_cur.cpp
rusdhist_srcs = $(addprefix $(RUSDHIST)/src/, $(rusdhist_cpplist))
rusdhist_objs	= ${rusdhist_srcs:.cpp=.o}

SDBINS += $(SDBINDIR)/rusdhist.run
.PHONY: rusdhist
rusdhist: $(SDBINDIR)/rusdhist.run
all:: rusdhist

# how to build the rusdhist.run program
$(SDBINDIR)/rusdhist.run: $(rusdhist_objs) ; \
$(LD) $(LDFLAGS) $^ $(rusdhist_libs) -o $@ ; \

clean:: ; \
rm -f $(RUSDHIST)/*~ $(RUSDHIST)/src/*~ $(rusdhist_objs); \
