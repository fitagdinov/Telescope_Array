
INCS += -I$(SDITERATOR)/inc
sditerator_libs = -L$(SDLIBDIR) -lsden -lsduti $(ROOTLIBS_ALT) $(DSTLIBS) 
sditerator_cpplist = sditerator.cpp sditerator_util.cpp sditerator_cppanalysis.cpp
sditerator_srcs = $(addprefix $(SDITERATOR)/src/, $(sditerator_cpplist))
sditerator_objs = ${sditerator_srcs:.cpp=.o}

SDBINS += $(SDBINDIR)/sditerator.run
.PHONY: sditerator
sditerator: $(SDBINDIR)/sditerator.run
all:: sditerator

$(SDBINDIR)/sditerator.run:$(sditerator_objs) ; \
$(LD) $(LDFLAGS) $^ $(sditerator_libs) -o $@ ; \


clean:: ; \
rm -f $(SDITERATOR)/*~ $(sditerator_objs) $(SDITERATOR)/src/*~ ; \
