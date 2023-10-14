// Functions that can be used while the event display is running

// to repeat some activity with every _delay_sec_ seconds interval
// until ENTER is pressed
#define repeat(_delay_sec_) while (p1.continue_activity(_delay_sec_))

// Drawing Tyro analysis arrival direction
void Set_Draw_Tyro_Arrow(Bool_t flag_Draw_Tyro_Arrow = false)
{ Draw_Tyro_Arrow = flag_Draw_Tyro_Arrow; }
