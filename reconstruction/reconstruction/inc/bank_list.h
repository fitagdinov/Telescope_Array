/* $Source: /hires_soft/uvm2k/uti/bank_list.h,v $
 * $Log: bank_list.h,v $
 * Last modified: DI 20191214
 *
 * Revision 1.4  2001/03/14 20:33:38  reil
 * Added C++ #ifdef
 *
 * Revision 1.3  1995/11/13 22:45:48  jeremy
 * Added C-simplified versions of routines.
 *
 * Revision 1.2  1995/10/17  15:51:55  jeremy
 * Added cmp_bank_list function.
 *
 * Revision 1.1  1995/05/09  00:53:16  jeremy
 * Initial revision
 *
*/

/* This collection of functions handle lists of dst banks */

#define BANK_LIST_ERROR -1 /* value returned by any bank_list function on error */

integer4 new_bank_list_(integer4 *size);
/* returns the ID of a new bank list */

integer4 del_bank_list_(integer4 *list);
/* deletes a bank list. returns 0 */

integer4 clr_bank_list_(integer4 *list);
/* removes all banks from bank list. returns 0 */

integer4 cnt_bank_list_(integer4 *list);
/* returns number of banks in bank list */

integer4 tst_bank_list_(integer4 *list, integer4 *bank);
/* return one if bank in list, zero otherwise */

integer4 add_bank_list_(integer4 *list, integer4 *bank);
/* add bank to list. return number banks add */

integer4 rem_bank_list_(integer4 *list, integer4 *bank);
/* remove bank from list. returns number of banks removed */

integer4 sum_bank_list_(integer4 *list1, integer4 *list2);
/* add banks in list2 to list1. returns number of banks added */

integer4 dif_bank_list_(integer4 *list1, integer4 *list2);
/* remove banks in list2 from list1. returns number of banks removed */

integer4 cpy_bank_list_(integer4 *list1, integer4 *list2);
/* copys banks in list2 to list1. returns number of banks copied */

integer4 com_bank_list_(integer4 *list1, integer4 *list2);
/* removes banks in list1 that are not in list2. returns number of banks in list1 */

integer4 itr_bank_list_(integer4 *list, integer4 *n);
/* return nth bank in list, increment n */

integer4 cmp_bank_list_(integer4 *list1, integer4 *list2);
/* return number of banks in list1 also in list2 */

void dsc_bank_list_(integer4 *list, FILE *fp);
/* describe banks lists by printing bank names in the list using FILE pointer fp */


/* C / C++ - simplified versions of above routines */
#ifdef __cplusplus
extern "C" {
#endif
  integer4 newBankList(integer4 size);
  integer4 delBankList(integer4 list);
  integer4 clrBankList(integer4 list);
  integer4 cntBankList(integer4 list);
  integer4 tstBankList(integer4 list, integer4 bank);
  integer4 addBankList(integer4 list, integer4 bank);
  integer4 remBankList(integer4 list, integer4 bank);
  integer4 sumBankList(integer4 list1, integer4 list2);
  integer4 difBankList(integer4 list1, integer4 list2);
  integer4 cpyBankList(integer4 list1, integer4 list2);
  integer4 comBankList(integer4 list1, integer4 list2);
  integer4 itrBankList(integer4 list, integer4 *n);
  integer4 cmpBankList(integer4 list1, integer4 list2);
  void     dscBankList(integer4 list,  FILE *fp);
#ifdef __cplusplus
} /* end extern "C" */
#endif
