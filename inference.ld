SECTIONS
{
    . = 0x2000;
    .text : {
        * (.start);
        * (.text);
    }
    .data : {
        * (.data);
        * (.rodata);
        * (.rodata.*);
        * (.sdata);
    }
    .bss : {
        * (.bss);
        * (.sbss);
    }
    _end = .;
}