
INCS += -I$(NUF)/inc
nuf_libs = -L$(SDLIBDIR) -lsden -lsduti -lgsl -lgslcblas
nuf_libs += $(ROOTLIBS_ALT) $(DSTLIBS) 

nuf_cpplist =  nuf.cpp cmdoptions.cpp iterate.cpp
nuf_cpplist +=  fit.cpp sdparamborder.cpp sign.cpp
nuf_srcs = $(addprefix $(NUF)/src/, $(nuf_cpplist))
nuf_objs = ${nuf_srcs:.cpp=.o}

SDBINS += $(SDBINDIR)/nuf.i12f.run

.PHONY: nuf
nuf: $(SDBINDIR)/nuf.i12f.run
all:: nuf

$(SDBINDIR)/nuf.i12f.run:$(nuf_objs) ; \
$(LD) $(LDFLAGS) $^ $(nuf_libs) -o $@ ; \

clean:: ; \
rm -f $(nuf_objs) $(NUF)/*~  $(NUF)/src/*~; \
