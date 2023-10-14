
# creating the temporary directory if it doesn't exist
pass1plot_tmp = $(PASS1PLOT)/tmp
$(if $(shell mkdir -p $(pass1plot_tmp) && test -d $(pass1plot_tmp) || echo "failed"),\
	$(error Failed to create the directory $(pass1plot_tmp)),)

INCS += -I$(PASS1PLOT)/inc
pass1plot_libs = -L$(SDLIBDIR) -lsdfdrt -lsduti -lsden $(ROOTLIBS) $(DSTLIBS)
pass1plot_src = $(PASS1PLOT)/src/pass1plot.cxx
pass1plot_src += $(PASS1PLOT)/src/pass1plot_misc.cxx
pass1plot_src += $(PASS1PLOT)/src/pass1plot_fd.cxx
pass1plot_src += $(PASS1PLOT)/src/pass1plot_debugging.cxx
pass1plot_src += $(pass1plot_tmp)/pass1plotDict.cxx
pass1plot_src += $(PASS1PLOT)/src/pass1plot_event_time_stamp.cxx
pass1plot_objs = ${pass1plot_src:.cxx=.o}


SDSOLIBS += $(SDLIBDIR)/libpass1plot.so
SDSASOLIBS += $(SDLIBDIR)/libpass1plot.so
.PHONY: pass1plot
pass1plot: $(SDLIBDIR)/libpass1plot.so
all:: pass1plot

$(SDLIBDIR)/libpass1plot.so: $(pass1plot_objs) ; \
$(LD) $(OPTOPT) -shared $^ $(pass1plot_libs) -o $@ ; \
find $(pass1plot_tmp) -name "*.pcm" -exec mv {} $(SDLIBDIR)/. \;

$(pass1plot_tmp)/pass1plotDict.cxx: $(PASS1PLOT)/inc/pass1plot.h $(PASS1PLOT)/inc/pass1plotLinkDef.h ; \
rootcint -f $@ $(ROOTCINTFLAGS) $(INCS) $(<F) $(filter %LinkDef.h, $^)

clean:: ; \
rm -f $(PASS1PLOT)/*~ $(PASS1PLOT)/src/*.o $(PASS1PLOT)/src/*~ ; \
rm -rf $(pass1plot_tmp)
