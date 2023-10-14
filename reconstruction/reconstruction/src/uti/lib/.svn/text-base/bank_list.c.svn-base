/* $Source: /hires_soft/uvm2k/uti/bank_list.c,v $
 * $Log: bank_list.c,v $
 *
 * Last modified: DI 20171206
 *
 * Revisian 1.6 2011/06/21 21:00 DI
 * Bug fixes 
 * 1) com_bank_list_ would sometimes remove all the banks in the list1
 * and not just the banks that are not in list2 because com_bank_list_
 * did not properly account for the fact that rem_bank_list changes
 * the contents of the bank_list array.  Fixed it by supplying an
 * additional variable to temporarily store the bank id that's being
 * considered for the removal.
 * 2) com_bank_list_ now returns the number of banks in list1 after
 * banks that are not in list2 have been removed, as the comments in
 * the function body and in the header file say it should.
 *
 * Revision 1.5  1996/01/24 08:53:08  mjk
 * Bug fixes
 * dif_bank_list_ and com_bank_list_ each need a variable initialized to 0
 * dif_bank_list_ and sum_bank_list_ need extra () in the if statement
 * which makes a call to rem_bank_list_. Otherwise they do not return
 * the proper values.
 *
 * Revision 1.4  1995/11/13  22:45:23  jeremy
 * Added C-simplified versions of routines.
 *
 * Revision 1.3  1995/10/17  15:51:53  jeremy
 * Added cmp_bank_list function.
 *
 * Revision 1.2  1995/05/10  01:00:52  jeremy
 * fixed bugs
 *
 * Revision 1.1  1995/05/09  00:53:16  jeremy
 * Initial revision
 *
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dst_std_types.h"
#include "bank_list.h"
#include "dst_size_limits.h"

#ifndef event_name_from_id_
integer4 event_name_from_id_(integer4 *bank_id, integer1 *name, integer4 *len);
#endif
#ifndef event_version_from_id_
integer4 event_version_from_id_(integer4 *bank_id);
#endif


#define BANK_LIST_INVALID(n) (n < 0 || n >= MAX_DST_BANK_LISTS || bank_list[n].bank == 0)

static struct
{
  integer4 size, count;
  integer4 *bank;
} bank_list[MAX_DST_BANK_LISTS];

/* This collection of functions handle lists of dst banks */

integer4 new_bank_list_(integer4 *size)
{
  /* returns the ID of a new bank list */
  int i;
  for (i = 0; i < MAX_DST_BANK_LISTS; ++i)
    if (bank_list[i].bank == 0)
      {
	bank_list[i].size = *size;
	bank_list[i].count = 0;
	if ( (bank_list[i].bank = calloc(*size, sizeof(integer4))) )
	  return i;
	break;
      }
  return BANK_LIST_ERROR;
}

integer4 newBankList(integer4 size)
{ return new_bank_list_(&size); }

integer4 del_bank_list_(integer4 *list)
{
  /* deletes bank list */
  if (BANK_LIST_INVALID(*list)) return BANK_LIST_ERROR;
  free(bank_list[*list].bank);
  bank_list[*list].bank = 0;
  return 0;
}

integer4 delBankList(integer4 list)
{ return del_bank_list_(&list); }

integer4 clr_bank_list_(integer4 *list)
{
  /* removes all banks from bank list */
  int count;
  if (BANK_LIST_INVALID(*list)) return BANK_LIST_ERROR;
  count = bank_list[*list].count;
  bank_list[*list].count = 0;
  return count;
}

integer4 clrBankList(integer4 list)
{ return clr_bank_list_(&list); }

integer4 cnt_bank_list_(integer4 *list)
{
  /* returns number of banks in bank list */
  if (BANK_LIST_INVALID(*list)) return BANK_LIST_ERROR;
  return bank_list[*list].count;
}

integer4 cntBankList(integer4 list)
{ return cnt_bank_list_(&list); }

integer4 tst_bank_list_(integer4 *list, integer4 *bank)
{
  /* return one if bank in list, zero otherwise */
  int i;
  if (BANK_LIST_INVALID(*list)) return BANK_LIST_ERROR;
  for (i = 0; i < bank_list[*list].count; ++i)
    if (bank_list[*list].bank[i] == *bank) return 1;
  return 0;
}

integer4 tstBankList(integer4 list, integer4 bank)
{ return tst_bank_list_(&list, &bank); }

integer4 add_bank_list_(integer4 *list, integer4 *bank)
{
  /* add bank to list. return number banks add or error */
  int i;
  if (BANK_LIST_INVALID(*list)) return BANK_LIST_ERROR;
  for (i = 0; i < bank_list[*list].count; ++i)
    if (bank_list[*list].bank[i] == *bank) return 0;
  if (i >= bank_list[*list].size - 1) return BANK_LIST_ERROR;
  bank_list[*list].bank[i++] = *bank;
  bank_list[*list].count = i;
  return 1;
}

integer4 addBankList(integer4 list, integer4 bank)
{ return add_bank_list_(&list, &bank); }

integer4 rem_bank_list_(integer4 *list, integer4 *bank)
{
  /* remove bank from list. returns number of banks removed or error */
  int i, j, count;
  if (BANK_LIST_INVALID(*list)) return BANK_LIST_ERROR;
  for (count = 0; ; --bank_list[*list].count, ++count)
    {
      for (i = 0; i < bank_list[*list].count; ++i)
	if (bank_list[*list].bank[i] == *bank) break;
      if (i >= bank_list[*list].count) return count;
      for (j = i + 1; j < bank_list[*list].count; i = j++)
	bank_list[*list].bank[i] = bank_list[*list].bank[j];
    }
}

integer4 remBankList(integer4 list, integer4 bank)
{ return rem_bank_list_(&list, &bank); }

integer4 sum_bank_list_(integer4 *list1, integer4 *list2)
{
  /* add banks in list2 to list1. returns number of banks added or error */
  int i, j, count;
  if (BANK_LIST_INVALID(*list1) || BANK_LIST_INVALID(*list2))
    return BANK_LIST_ERROR;
  for (count = i = 0; i < bank_list[*list2].count; count += j, ++i)
    if ((j = add_bank_list_(list1, &bank_list[*list2].bank[i])) == 
	BANK_LIST_ERROR)
      return BANK_LIST_ERROR;
  return count;
}

integer4 sumBankList(integer4 list1, integer4 list2)
{ return sum_bank_list_(&list1, &list2); }

integer4 dif_bank_list_(integer4 *list1, integer4 *list2)
{
  /* remove banks in list2 from list1. returns number of banks 
     removed or error */
  int i, j=0, count;
  if (BANK_LIST_INVALID(*list1) || BANK_LIST_INVALID(*list2))
    return BANK_LIST_ERROR;
  for (count = i = j; i < bank_list[*list2].count; count += j, ++i)
    if ((j = rem_bank_list_(list1, &bank_list[*list2].bank[i])) == 
	BANK_LIST_ERROR)
      return BANK_LIST_ERROR;
  return count;
}

integer4 difBankList(integer4 list1, integer4 list2)
{ return dif_bank_list_(&list1, &list2); }

integer4 cpy_bank_list_(integer4 *list1, integer4 *list2)
{
  /* copys banks in list2 to list1. returns number of banks copied or error */
  int count;
  if (clr_bank_list_(list1) == BANK_LIST_ERROR ||
      (count = sum_bank_list_(list1, list2)) == BANK_LIST_ERROR)
    return BANK_LIST_ERROR;
  return count;
}

integer4 cpyBankList(integer4 list1, integer4 list2)
{ return cpy_bank_list_(&list1, &list2); }

integer4 com_bank_list_(integer4 *list1, integer4 *list2)
{
  /* removes banks in list1 that are not in list2. returns number of 
     banks in list1 or error */
  int i,bank;
  if (BANK_LIST_INVALID(*list1) || BANK_LIST_INVALID(*list2))
    return BANK_LIST_ERROR;
  for (i = 0; i < bank_list[*list1].count; ++i)
    {
      if (tst_bank_list_(list2,&bank_list[*list1].bank[i])==0)
	{
	  /* rem_bank_list changes the bank_list contents in an iterative way, 
	     so first save the bank id of the bank being removed */
	  bank = bank_list[*list1].bank[i];
	  rem_bank_list_(list1,&bank);
	  /* to re-iterate the current i-value, because bank_list moved down by 1 */
	  --i;
	}
    }
  return bank_list[*list1].count;
}

integer4 comBankList(integer4 list1, integer4 list2)
{ return com_bank_list_(&list1, &list2); }

integer4 itr_bank_list_(integer4 *list, integer4 *n)
{
  /* return nth bank in list, increment n */
  int i;
  if (BANK_LIST_INVALID(*list)) return BANK_LIST_ERROR;
  if ((i = *n) >= bank_list[*list].count) return 0;
  return *n += 1, bank_list[*list].bank[i];
}

integer4 itrBankList(integer4 list, integer4 *n)
{ return itr_bank_list_(&list, n); }

integer4 cmp_bank_list_(integer4 *list1, integer4 *list2)
{
  /* return number of banks in list1 also in list2 */
  int i, cnt = 0;
  if (BANK_LIST_INVALID(*list1) || BANK_LIST_INVALID(*list2))
    return BANK_LIST_ERROR;
  for (i = 0; i < bank_list[*list1].count; ++i)
    if (tst_bank_list_(list2, &bank_list[*list1].bank[i])) ++cnt;
  return cnt;
}

integer4 cmpBankList(integer4 list1, integer4 list2)
{ return cmp_bank_list_(&list1, &list2); }


void dsc_bank_list_(integer4 *list, FILE *fp)
{
  integer4 rc,len,type,event,sum=0;
  char text[0x100];
  integer4 size = 0x100;
  for (rc = len = 0; (type = itr_bank_list_(list, &rc)) > 0; len += event)
    {
      event_name_from_id_(&type, text, &size);
      sum += event_version_from_id_(&type);
      event = 2 + strlen(text);
      if (len + event > 78) len = 0;
      if (rc)
	{
	  if (len)
	    {
	      fputc(',', fp);
	      fputc(' ', fp);
	    }
	  else fputc('\n', fp);
	}
      if (!len) fputs("  ", fp);
      fputs(text, fp);
    }
  fprintf(fp,"\n  %d banks total", cnt_bank_list_(list));
  fprintf(fp,"\n  bank version sum: %d\n",sum);
}

void dscBankList(integer4 list, FILE *fp)
{ return dsc_bank_list_(&list,fp); }
