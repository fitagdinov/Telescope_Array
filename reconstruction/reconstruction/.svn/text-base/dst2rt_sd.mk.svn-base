
INCS += -I$(DST2RT_SD)/inc
dst2rt_sd_libs = -L$(SDLIBDIR) -lsdfdrt -lsduti $(ROOTLIBS_ALT) $(DSTLIBS)
dst2rt_sd_cpplist	= dst2rt_sd.cpp dst2rt_sd_util.cpp
dst2rt_sd_srcs = $(addprefix $(DST2RT_SD)/src/, $(dst2rt_sd_cpplist))
dst2rt_sd_objs	= ${dst2rt_sd_srcs:.cpp=.o}

SDBINS += $(SDBINDIR)/dst2rt_sd.run
.PHONY: dst2rt_sd
dst2rt_sd: $(SDBINDIR)/dst2rt_sd.run
all:: dst2rt_sd

$(SDBINDIR)/dst2rt_sd.run: $(dst2rt_sd_objs) ; \
$(LD) $(LDFLAGS) $^ $(dst2rt_sd_libs) -o $@ ; \

clean:: ; \
rm -f $(dst2rt_sd_objs) $(DST2RT_SD)/*~ $(DST2RT_SD)src/*~ ; \
