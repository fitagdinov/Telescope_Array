
sdscript_list = sdrec sdrec_parallel sd_prep4hb sded
sdscript_list += sdmc_run_sdmc_calib_extract sdmc_run_sdmc_calib_check
sdscript_list += sdmc_prep_sdmc_run sdmc_showlib_info sdmc_cmb_tothrow_files  
sdscript_list += sdmc_run_sdmc_spctr sdmc_run_sdmc_conv_e2_to_hires_spctr 
sdscript_list += sdmc_run_sdmc_conv_e2_to_spctr sdmc_calib_check2rt
sdscript_list += sdmc_run_sdmc_tsort sdmc_run_sdmc_byday


sdscript_files = $(addprefix $(SDBINDIR)/, $(sdscript_list))

.PHONY: sdscripts
sdscripts: $(sdscript_files)
all:: sdscripts

clean:: ; rm -f $(sdscript_files)

# SD Event reconstruction, full chain
$(SDBINDIR)/sdrec: $(SDSCRIPTS)/sdrec_script ; \
echo "#!/usr/bin/env bash"> $@ ; \
echo "">>$@ ; \
echo "# Automatically generated file.  DO NOT EDIT - all changes will be lost after another build ">>$@; \
echo "# Instead, edit $^ ">>$@; \
cat $^ | grep -v '#!/usr/bin' >>$@ ; \
chmod u+x $@

# SD Event reconstruction, full chain, parallelized
$(SDBINDIR)/sdrec_parallel: $(SDSCRIPTS)/sdrec_parallel_script ; \
echo "#!/usr/bin/env bash"> $@ ; \
echo "">>$@ ; \
echo "# Automatically generated file.  DO NOT EDIT - all changes will be lost after another build ">>$@; \
echo "# Instead, edit $^ ">>$@; \
cat $^ | grep -v '#!/usr/bin' >>$@ ; \
chmod u+x $@

# To Prepare SD data for conventional hybrid analyses
# that do not use the hybrid trigger
$(SDBINDIR)/sd_prep4hb: $(SDSCRIPTS)/sd_prep4hb ; \
echo "#!/usr/bin/env bash"> $@ ; \
echo "">>$@ ; \
echo "# Automatically generated file.  DO NOT EDIT - all changes will be lost after another build ">>$@; \
echo "# Instead, edit $^ ">>$@; \
cat $^ | grep -v '#!/usr/bin' >>$@ ; \
chmod u+x $@

# TA SD Event display that runs on detailed root files
# that are produced by the sdrec sdscript.
$(SDBINDIR)/sded: $(SDSCRIPTS)/sded_script ; \
echo "#!/usr/bin/env bash"> $@ ; \
echo "">>$@ ; \
echo "# Automatically generated file.  DO NOT EDIT - all changes will be lost after another build ">>$@; \
echo "# Instead, edit $^ ">>$@; \
cat $^ | grep -v '#!/usr/bin' >>$@ ; \
chmod u+x $@


# SD Monte Carlo scripts

$(SDBINDIR)/sdmc_run_sdmc_calib_extract: $(SDSCRIPTS)/sdmc_run_sdmc_calib_extract_script; \
echo "#!/usr/bin/env bash"> $@ ; \
echo "">>$@ ; \
echo "# Automatically generated file.  DO NOT EDIT - all changes will be lost after another build ">>$@; \
echo "# Instead, edit $^ ">>$@; \
cat $^ | grep -v '#!/usr/bin' >>$@ ; \
chmod u+x $@

$(SDBINDIR)/sdmc_run_sdmc_calib_check: $(SDSCRIPTS)/sdmc_run_sdmc_calib_check_script; \
echo "#!/usr/bin/env bash"> $@ ; \
echo "">>$@ ; \
echo "# Automatically generated file.  DO NOT EDIT - all changes will be lost after another build ">>$@; \
echo "# Instead, edit $^ ">>$@; \
cat $^ | grep -v '#!/usr/bin' >>$@ ; \
chmod u+x $@


$(SDBINDIR)/sdmc_run_sdmc_spctr: $(SDSCRIPTS)/sdmc_run_sdmc_spctr_script; \
echo "#!/usr/bin/env bash"> $@ ; \
echo "">>$@ ; \
echo "# Automatically generated file.  DO NOT EDIT - all changes will be lost after another build ">>$@; \
echo "# Instead, edit $^ ">>$@; \
cat $^ | grep -v '#!/usr/bin' >>$@ ; \
chmod u+x $@

$(SDBINDIR)/sdmc_run_sdmc_conv_e2_to_spctr: $(SDSCRIPTS)/sdmc_run_sdmc_conv_e2_to_spctr_script; \
echo "#!/usr/bin/env bash"> $@ ; \
echo "">>$@ ; \
echo "# Automatically generated file.  DO NOT EDIT - all changes will be lost after another build ">>$@; \
echo "# Instead, edit $^ ">>$@; \
cat $^ | grep -v '#!/usr/bin' >>$@ ; \
chmod u+x $@

$(SDBINDIR)/sdmc_run_sdmc_conv_e2_to_hires_spctr: $(SDSCRIPTS)/sdmc_run_sdmc_conv_e2_to_hires_spctr_script; \
echo "#!/usr/bin/env bash"> $@ ; \
echo "">>$@ ; \
echo "# Automatically generated file.  DO NOT EDIT - all changes will be lost after another build ">>$@; \
echo "# Instead, edit $^ ">>$@; \
cat $^ | grep -v '#!/usr/bin' >>$@ ; \
chmod u+x $@

$(SDBINDIR)/sdmc_showlib_info: $(SDSCRIPTS)/sdmc_showlib_info_py_script; \
echo "#!/usr/bin/env python"> $@ ; \
echo "">>$@ ; \
echo "# Automatically generated file.  DO NOT EDIT - all changes will be lost after another build ">>$@; \
echo "# Instead, edit $^ ">>$@; \
cat $^ | grep -v '#!/usr/bin' >>$@ ; \
chmod u+x $@

$(SDBINDIR)/sdmc_calib_check2rt: $(SDSCRIPTS)/sdmc_calib_check2rt_py_script; \
echo "#!/usr/bin/env python"> $@ ; \
echo "">>$@ ; \
echo "# Automatically generated file.  DO NOT EDIT - all changes will be lost after another build ">>$@; \
echo "# Instead, edit $^ ">>$@; \
cat $^ | grep -v '#!/usr/bin' >>$@ ; \
chmod u+x $@

$(SDBINDIR)/sdmc_prep_sdmc_run: $(SDSCRIPTS)/sdmc_prep_sdmc_run_py_script; \
echo "#!/usr/bin/env python"> $@ ; \
echo "">>$@ ; \
echo "# Automatically generated file.  DO NOT EDIT - all changes will be lost after another build ">>$@; \
echo "# Instead, edit $^ ">>$@; \
cat $^ | grep -v '#!/usr/bin' >>$@ ; \
chmod u+x $@

$(SDBINDIR)/sdmc_cmb_tothrow_files: $(SDSCRIPTS)/sdmc_cmb_tothrow_files_py_script; \
echo "#!/usr/bin/env python"> $@ ; \
echo "">>$@ ; \
echo "# Automatically generated file.  DO NOT EDIT - all changes will be lost after another build ">>$@; \
echo "# Instead, edit $^ ">>$@; \
cat $^ | grep -v '#!/usr/bin' >>$@ ; \
chmod u+x $@

$(SDBINDIR)/sdmc_run_sdmc_tsort: $(SDSCRIPTS)/sdmc_run_sdmc_tsort_script; \
echo "#!/usr/bin/env bash"> $@ ; \
echo "">>$@ ; \
echo "# Automatically generated file.  DO NOT EDIT - all changes will be lost after another build ">>$@; \
echo "# Instead, edit $^ ">>$@; \
cat $^ | grep -v '#!/usr/bin' >>$@ ; \
chmod u+x $@

$(SDBINDIR)/sdmc_run_sdmc_byday: $(SDSCRIPTS)/sdmc_run_sdmc_byday_script; \
echo "#!/usr/bin/env bash"> $@ ; \
echo "">>$@ ; \
echo "# Automatically generated file.  DO NOT EDIT - all changes will be lost after another build ">>$@; \
echo "# Instead, edit $^ ">>$@; \
cat $^ | grep -v '#!/usr/bin' >>$@ ; \
chmod u+x $@
