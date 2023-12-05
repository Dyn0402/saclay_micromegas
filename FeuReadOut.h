
#ifndef FeuReadOut_h
#define FeuReadOut_h

// #include <TXNetFile.h>
#include <TFile.h>
#include "dreamdataline.h"

class FeuReadOut {

 public:
  int Id;
  int N;
  TFile* dfile;
  DataLineDream current_data;
  bool data_to_treat;
  bool event_completed;
  bool FeuHeaderLoaded;
  int FeuHeaderLine;
  int DataHeaderLine;
  int asicN;
  int detN;
  int channelN;
  int EventID;
  int TimeStamp;
  int FineTimeStamp;
  int isample;
  int isample_prev;
  bool zs_mode;

  FeuReadOut(TFile* df=0) : Id(-1), N(-1), dfile(df), data_to_treat(false), event_completed(false),
                             FeuHeaderLine(0), DataHeaderLine(0), asicN(0), detN(0), channelN(0),
                             EventID(-1), TimeStamp(-1),FineTimeStamp(-1), isample(-1), isample_prev(-2), zs_mode(false)  {  }
  void NewEvent() { 
    FeuHeaderLoaded=false; 
    EventID=-1; 
    event_completed=false; 
    isample=-1; 
    isample_prev=-2;
    FineTimeStamp = -1;
    FeuHeaderLine=0; 
    DataHeaderLine=0;  }
};




#endif
