
INCS += -I$(SDASCII)/inc
sdascii_libs = -L$(SDLIBDIR) -lsden -lsduti $(ROOTLIBS_ALT) $(DSTLIBS)

sdascii_cpplist = sdascii.cpp sdascii_util.cpp
sdascii_srcs = $(addprefix $(SDASCII)/src/, $(sdascii_cpplist))
sdascii_objs	= ${sdascii_srcs:.cpp=.o}

SDBINS += $(SDBINDIR)/sdascii.run
all:: $(SDBINDIR)/sdascii.run

$(SDBINDIR)/sdascii.run: $(sdascii_objs) ; \
$(LD) $(LDFLAGS) $^ $(sdascii_libs) -o $@ ; \

clean:: ; \
rm -f $(sdascii_objs) $(SDASCII)/*~ $(SDASCII)/src/*~ ; \
