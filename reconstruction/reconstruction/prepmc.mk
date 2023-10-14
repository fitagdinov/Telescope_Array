
INCS += -I$(PREPMC)/inc
prepmc_libs = -L$(SDLIBDIR) -lsdfdrt -lsduti $(ROOTLIBS_ALT) $(DSTLIBS)
prepmc_cpplist	= prepmc.cpp prepmc_util.cpp
prepmc_srcs = $(addprefix $(PREPMC)/src/, $(prepmc_cpplist))
prepmc_objs	= ${prepmc_srcs:.cpp=.o}

SDBINS += $(SDBINDIR)/prepmc.run
.PHONY: prepmc
prepmc: $(SDBINDIR)/prepmc.run
all:: prepmc

$(SDBINDIR)/prepmc.run: $(prepmc_objs) ; \
$(LD) $(LDFLAGS) $^ $(prepmc_libs) -o $@ ; \

clean:: ; \
rm -f $(prepmc_objs) $(PREPMC)/*~ $(PREPMC)src/*~ ; \
