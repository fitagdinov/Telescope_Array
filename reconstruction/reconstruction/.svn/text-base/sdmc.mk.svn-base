
INCS += -I$(SDMC)

LIBS = -L$(SDLIBDIR) -lsduti -L$(DSTDIR)/lib -ldst2k -lm  -lc -lz -lbz2

# default is gcc
sdmc_spctr_suf=gcc_x86
chk32bit=$(shell uname -a | grep i686)
ifneq (x$(chk32bit),x)
sdmc_spctr_suf=gcc_i686
endif

# If intel compiler is used and we are on a CHPC platform
ifneq (x$(usingicc),x)
sdmc_spctr_suf=icc_x86
ifneq (x$(chk32bit),x)
sdmc_spctr_suf=icc_i686
endif
ifeq ($(UUFSCELL),kingspeak.peaks)
sdmc_spctr_suf:=icc_kp
endif
ifeq ($(UUFSCELL),ember.arches)
sdmc_spctr_suf:=icc_emb
endif
ifeq ($(UUFSCELL),ash.peaks)
sdmc_spctr_suf:=icc_ash
endif
ifeq ($(UUFSCELL),lonepeak.peaks)
sdmc_spctr_suf:=icc_lnp
endif
endif

sdmc_c_binlist  = sdmc_calib_extract.run
sdmc_c_binlist  += sdmc_calib_check.run
sdmc_c_binlist  += sdmc_calib_show.run
sdmc_c_binlist  += sdmc_conv_e2_to_spctr.run
sdmc_c_binlist  += sdmc_spctr_2g_$(sdmc_spctr_suf).run
sdmc_c_bins = $(addprefix $(SDBINDIR)/, $(sdmc_c_binlist))

sdmc_cpp_binlist  = bsdinfo.run
sdmc_cpp_binlist  += sdmc_tsort.run
sdmc_cpp_binlist  += sdmc_byday.run
sdmc_cpp_bins = $(addprefix $(SDBINDIR)/, $(sdmc_cpp_binlist))

.PHONY: sdmc
sdmc: $(sdmc_c_bins) $(sdmc_cpp_bins)
all:: sdmc

clean:: ; \
rm -f $(sdmc_c_bins) $(sdmc_cpp_bins) $(SDMC)/*.o


$(sdmc_c_bins) : $(SDBINDIR)/%.run : $(SDMC)/%.o; \
$(CC) $(CFLAGS) $(INCS) -o $@ $^ $(LIBS)

$(sdmc_cpp_bins) : $(SDBINDIR)/%.run : $(SDMC)/%.o; \
$(CPP) $(CPPFLAGS) $(INCS) -o $@ $^ $(LIBS)


SDBINS += $(sdmc_c_bins) $(sdmc_cpp_bins)

########################### SDMC-SPECIFIC SUFFIX RULES #########################

%_1g_$(sdmc_spctr_suf).o : %.c ; \
$(CC) -DNT=1 $(CFLAGS) $(INCS) -o $@ $< -c

%_2g_$(sdmc_spctr_suf).o : %.c ; \
$(CC) -DNT=20 $(CFLAGS) $(INCS) -o $@ $< -c

%_3g_$(sdmc_spctr_suf).o : %.c ; \
$(CC) -DNT=25 -mcmodel=medium $(CFLAGS) $(INCS) -o $@ $< -c

%_4g_$(sdmc_spctr_suf).o : %.c ; \
$(CC) -DNT=50 -mcmodel=medium $(CFLAGS) $(INCS) -o $@ $< -c

%_8g_$(sdmc_spctr_suf).o : %.c ; \
$(CC) -DNT=48 -mcmodel=medium $(CFLAGS) $(INCS) -o $@ $< -c

%_16g_$(sdmc_spctr_suf).o : %.c ; \
$(CC) -DNT=96 -mcmodel=medium $(CFLAGS) $(INCS) -o $@ $< -c
