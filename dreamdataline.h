#ifndef dreamdataline_h
#define dreamdataline_h

class DataLineDream {
	public:
		DataLineDream()  { data = 0; }
		~DataLineDream()  {  }
		void ntohs_(){ data = ntohs(data); };
		bool is_final_trailer() const  { return (((data) & 0x7000)>>12)==7; }  // X111
		bool is_data_trailer() const  { return (((data) & 0x6000)>>13)==2; }  // X10X
		bool is_first_line() const  { return (((data) & 0x7000)>>12)==3; }  // X011
		bool is_data() const  { return (((data) & 0x7000)>>12)==0; }  // X000
		bool is_data_zs() const  { return (((data) & 0x6000)>>13)==0; }  // X00X
		bool is_channel_ID() const  { return (((data) & 0x7000)>>12)==1; }  // X001
		bool is_Feu_header() const  { return (((data) & 0x7000)>>12)==6; }  // X110
		bool is_data_header() const  { return (((data) & 0x6000)>>13)==1; }  // X01X
		bool get_zs_mode() const  { return (((data) & 0x400)>>10); }
		int get_Feu_ID() const  { return (((data) & 0xFF)); }
		int get_sample_ID() const  { return (((data) & 0xFF8)>>3); }//#define GET_SAMPLE_INDEX(word)  ((word & 0x0FF8)>>3)
		int get_finetstp() const  { return (((data) & 0x0007)); }//#define GET_FINETSTP(word)      (word & 0x0007)
		int get_channel_ID() const  { return (((data) & 0x3F)); }
		int get_dream_ID() const  { return (((data) & 0xE00)>>9); }
		int get_data() const  { return (((data) & 0xFFF)); }
		unsigned short int data;
};


#endif
