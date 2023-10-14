dstbanks_mk = $(DSTDIR)/dstbanks.mk
$(if $(wildcard $(dstbanks_mk)),,\
        $(error $(dstbanks_mk) not found))
include $(dstbanks_mk)
dst2k_srcbank = $(addsuffix _dst.c, $(dstbanks))
dst2k_bank    = $(addprefix $(DSTDIR)/src/bank/lib/,${dst2k_srcbank:.c=.o})
dst2k_srcdst  = $(wildcard $(DSTDIR)/src/dst/lib/*.c)
dst2k_dst     = ${dst2k_srcdst:.c=.o}
dst2k_srcuti  = $(wildcard $(DSTDIR)/src/uti/lib/*.c)
dst2k_uti     = ${dst2k_srcuti:.c=.o}
dst2k_objs    = $(dst2k_bank) $(dst2k_dst) $(dst2k_uti)
dst2k_binsrc  = $(wildcard $(DSTDIR)/src/*.c)
dst2k_binobj  = ${dst2k_binsrc:.c=.o}
dst2k_bins    = $(subst $(DSTDIR)/src,$(SDBINDIR),${dst2k_binobj:.o=.run})
dst2k_libs    = $(DSTLIBS)
SDLIBS       += $(DSTDIR)/lib/libdst2k.a
SDBINS       += $(dst2k_bins)
.PHONY: dst2k-ta
dst2k-ta: $(DSTDIR)/lib/libdst2k.a $(dst2k_bins)
all:: dst2k-ta
$(DSTDIR)/lib/libdst2k.a: $(dst2k_objs) ;\
$(AR) rcs $@ $?
clean:: ; rm -f $(dst2k_objs) $(dst2k_binobj)
$(dst2k_bins): $(SDBINDIR)/%.run: $(DSTDIR)/src/%.o $(DSTDIR)/lib/libdst2k.a ;\
$(LD) $(LDFLAGS) $^ $(dst2k_libs) -o $@
