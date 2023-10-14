#ifndef _sdanalysis_icc_settings_h_
#define _sdanalysis_icc_settings_h_

#ifdef __INTEL_COMPILER

/* Ignore icc remark : "operands are evaluated in unspecified order"*/
#pragma warning(disable:981)

/* Ignore icc remark : "external function definition with no prior declaration" */
//#pragma warning(disable:1418)

/* Ignore icc remark : "external declaration in primary source file" */
//#pragma warning(disable:1419)

/* Ignore icc remark : " parameter "arg" was never referenced" */
#pragma warning(disable:869)

#endif

#endif
