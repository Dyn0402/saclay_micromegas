// CEA de Saclay
// DSM - IRFU - SPhN
// CLAS12 Group
// Code for Data Analysis of Micromegas prototype of CLAS12

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <time.h>
#include <string.h>
#include <signal.h>

#include <TMath.h>
#include <TROOT.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TF1.h>
#include <TTree.h>
// #include <TXNetFile.h>
#include <TFile.h>
#include "TCanvas.h"
#include "TApplication.h"
#include <TSystem.h>
#include <TError.h>

#include <iostream>
#include <list>
#include <map>
#include <vector>
#include <string>

// #include "mygblink.h"
#include "dreamdataline.h"
#include "FeuReadOut.h"


using namespace std;

Int_t COMPRESS=0; // 0: compress false; 1: compress true

const int Nfeu=1;
const int Ndet=8*Nfeu; // number of dreams in data
const Int_t NstripMax=64; // number of strips per asic
const Int_t LowerCut=0; //Sample min
const Int_t HigherCut=32; //Sample max
const Int_t Nsample=HigherCut-LowerCut;//number of sample
Double_t MaxThreshold=4000;
int Version; // version of the daq
Int_t compteur;
int sampleCountMax;

FILE* f;
list<string> dataFileNames;

TH1F *h0 = new TH1F("h0","Not Ploted, intermediaire of calcul.",500,-1000,1000);
// variable of the output root tree
int Nevent = 0, Nbadevent = 0; // current event number
int IDEvent = 0;  // ID of event in Feu header
int FineTimeStamp[Nfeu];  // Fine Time Stamp per feu

Bool_t event_flag=kFALSE;
Bool_t sat_flag=kFALSE;
Bool_t select_flag=kFALSE;
Bool_t strip_flag[Ndet][NstripMax];
Bool_t compressMod_flag= (bool) COMPRESS;

void signal_handler(int sig);
struct sigaction sigactif;
bool stopPgm = false;



//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

class My_LorentzAngle {

 protected:
  int StripAmpl[Ndet][NstripMax][Nsample];
  int TsampleNum[Nsample]; // index of time sample
  int StripNum[NstripMax]; // index of strip
  vector<FeuReadOut> feux;
  map<int,int> feuIDToN;
  map<int,int> feuNToID;
  bool bad_event;  // flag to tag bad event

 public:
  My_LorentzAngle();
  virtual ~My_LorentzAngle();
  void ReadOnlineOld();
  void ReadOnline();
  void My_GetPicPulsePosition(Int_t,Int_t, Int_t);
  void CleanStripAmpl();
  void ReadFeuIDs(const char* filename);
  bool ReadFeuHeaders(FeuReadOut& feu);
  bool ReadDreamData(FeuReadOut& feu, bool skipping=false);
  bool ReadFeuTrailer(FeuReadOut& feu, bool skipping=false);
  bool SkipEvent(FeuReadOut& feu);
  bool ReadEvent(FeuReadOut& feu);
  int  gtEvID(int evid1, int evid2);
  Double_t My_WeightedAverageMethod(Int_t);
};



time_t td;
struct tm *dcp;


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

My_LorentzAngle::My_LorentzAngle() {
  feuIDToN.clear();
  feuNToID.clear();
  feux.clear();
  bad_event = false;
}


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void My_LorentzAngle::ReadFeuIDs(const char* filename) {
  char tmpstr[300];

  feuIDToN.clear();
  feuNToID.clear();

  FILE* fd = fopen(filename, "r");
  if (!fd) {
    cerr<<"Error in My_LorentzAngle::ReadFeuIDs: can't read "<<filename<<endl;
    return;
  }

  int n = -1, id = -1;
  while (!feof(fd)) {
    fgets(tmpstr, 300, fd);
    if (tmpstr[0]==0 || tmpstr[0]=='#') continue;
    register int res = sscanf(tmpstr, "%d %d", &n, &id);
    if (res == 2 && n >=0 && n < Ndet/8)  {
      cout<<"Feu ID "<<id<<" n� "<<n<<endl;
      feuIDToN[id] = n;
      feuNToID[n] = id;
    }
  }

  fclose(fd);
}


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

int My_LorentzAngle::gtEvID(int evid1, int evid2) {
  if (evid1 == evid2) return 0;
  register bool looped = (abs(evid1-evid2) > 2048);
  if (!looped) {
    if (evid1 > evid2) return 1;
    else return -1;
  } else {
    if (evid1 > evid2) return -1;
    else return 1;
  }
}


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void My_LorentzAngle::ReadOnline() {
  char rawFileName[300];
  char strtmp[200];

   TTree::SetMaxTreeSize(8000000000);  // to split tree file above 8GB
  // TTree::SetMaxTreeSize(1000000);  // to split tree file above 1GB

  // Open the file
    FILE *inputFile = fopen("dream_data_reader_output_file_name.txt", "r");

    // Check if the file is open
    if (inputFile == NULL) {
        fprintf(stderr, "Unable to open the file.\n");
    }

    // Read the filename from the file
    char filename[100]; // Adjust the size according to your needs
    fscanf(inputFile, "%s", filename);

    // Close the file
    fclose(inputFile);

    // Display the filename
    printf("Read filename: %s\n", filename);

  TFile *file = new TFile(filename,"RECREATE");
  file->cd();
  TTree *treer = new TTree("T","event");
  treer->Branch("Nevent", &Nevent, "Nevent/I"); // event number
  treer->Branch("IDEvent", &IDEvent, "IDEvent/I"); // ID of the event
  sprintf(strtmp,"FineTimeStamp[%d]/I",Nfeu);
  treer->Branch("FineTimeStamp", FineTimeStamp,strtmp); // Fine time stamp
  sprintf(strtmp,"StripAmpl[%d][%d][%d]/I",Ndet,NstripMax,Nsample);
  treer->Branch("StripAmpl", StripAmpl, strtmp); // raw amplitude
  sprintf(strtmp,"TsampleNum[%d]/I",Nsample);
  treer->Branch("TsampleNum", TsampleNum, strtmp); // time sample number
  sprintf(strtmp,"StripNum[%d]/I",NstripMax);
  treer->Branch("StripNum", StripNum, strtmp); // time sample number
  
  for (register Int_t strip=0; strip<NstripMax; strip++) {
    StripNum[strip] = strip;
    for (register Int_t itime=LowerCut; itime<HigherCut; itime++) {
      TsampleNum[itime-LowerCut] = itime;
    }
  }


  for (list<string>::iterator ifname=dataFileNames.begin(); ifname!=dataFileNames.end(); ifname++) {
    TFile* dfiletmp=0;
    sprintf(rawFileName,"%s?filetype=raw",(*ifname).c_str());
    if (memcmp(rawFileName,"root",4) == 0) {
//      dfiletmp = new TXNetFile(rawFileName,"READ");
    } else {
      dfiletmp = new TFile(rawFileName,"READ");
    }
    if (dfiletmp == NULL) {
      cout<<"fichier de donnees introuvable: "<<*ifname<<endl;
      exit(1);
    } else {
      feux.push_back(FeuReadOut(dfiletmp));
      cout<<"file "<<*ifname<<" loaded.."<< endl;
    }
  }

  register int Nfeux = feux.size();
  if(Nfeux!=Nfeu) cout << "Mismatch of nb of Feu" << endl;
  register int maxEvID=0, minEvID=0;
  register bool badreadfg = false;

  while (true) {  // loop on events

    // checks all headers and get EventIDs
    maxEvID=0; minEvID=0;
    for (register int ifeu=0; ifeu<Nfeux; ifeu++) {
      feux[ifeu].NewEvent();
      badreadfg = ReadFeuHeaders(feux[ifeu]);
      if (badreadfg) { cerr<<"Error in feu header reading for feu "<<ifeu<<endl; break; }
      if (ifeu == 0) {
        maxEvID = minEvID = feux[ifeu].EventID;
      } else {
        register int evId = feux[ifeu].EventID;
        if (gtEvID(evId, maxEvID) == 1) maxEvID = evId;
        if (gtEvID(evId, minEvID) == -1) minEvID = evId;
      }
    }
    if (gtEvID(minEvID, maxEvID) != 0) {  //events shifted, should skip desynch parts
      cerr<<"Desync event, skipping some files, minEvID "<<minEvID<<" maxEvID "<<maxEvID<<endl;
      cerr<<"  Feu concerned: ";
      for (register int ifeu=0; ifeu<Nfeux; ifeu++) {
        cerr<<ifeu<<" (feuN "<<feux[ifeu].N<<") ";
        if (feux[ifeu].EventID != maxEvID) {
          badreadfg = SkipEvent(feux[ifeu]);  // skip partial event in late files
          if (badreadfg) {
            cerr<<"  Error in event skipping for feu "<<ifeu<<" at event "<<Nevent<<endl;
            break;
          }
        }
      }
      cerr<<endl;
      if (badreadfg) {
        cerr<<"  Error in event skipping"<<endl;
        break;  // break from event loop
      }
      continue;  // try again after event skipping
    }

    for (register int ifeu=0; ifeu<Nfeux; ifeu++) {
      badreadfg = ReadEvent(feux[ifeu]);
      if (badreadfg) {
        cerr<<"  Error in event reading for feu "<<ifeu<<" at event "<<Nevent<<endl;
        break;
      }
    }
    if (badreadfg) { break; }  // break from event loop

    if ((Nevent%100) == 0) cerr << Nevent << " events processed in files, and "<<Nbadevent<<" skipped " << endl;
    if (!bad_event) {
      //cout << "Event : " << Nevent << endl;
      // for (register int ifeu=0; ifeu<Nfeux; ifeu++) {//here
      // 	FineTimeStamp[ifeu]=feux[ifeu].FineTimeStamp;
      // 	cout << "Feu["<< ifeu << "]=" << feux[ifeu].N << "\t" << feux[ifeu].FineTimeStamp << endl; 
      // }
      IDEvent = minEvID;  // to store in tree
      treer->Fill();
      Nevent++;
    } else {
      Nbadevent++;
    }
    CleanStripAmpl();
    for (register int ifeu=0; ifeu<Nfeux; ifeu++) FineTimeStamp[ifeu]=-1; //clean finetimestamp
    bad_event = false;
    if (stopPgm) break;
  }//event loop

  cout << "writing file..." << endl;
//   treer->FlushBaskets();
  file = treer->GetCurrentFile();  // in case of file splitting
  file->cd();
  treer->Write();
//   file->Flush();
  file->Close();
  cout << "file written!" << endl;
//   treer->Delete();
  cout<<" end reading -->Nevent "<<Nevent<<" and Nbadevent "<<Nbadevent<<endl;
  exit(1);

}


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

bool My_LorentzAngle::ReadFeuHeaders(FeuReadOut& feu) {
  register bool badreadfg;

  if (feu.FeuHeaderLoaded) return false;  // already done

  if (!feu.data_to_treat) {
    badreadfg = feu.dfile->ReadBuffer((char*)&feu.current_data, sizeof(feu.current_data));
    if (badreadfg) return true;  // failed
    feu.current_data.ntohs_(); feu.data_to_treat = true;
  }

  while (true) {
    if (feu.FeuHeaderLine<8 && feu.current_data.is_Feu_header()) {
      if (feu.FeuHeaderLine==0) {
	feu.zs_mode = feu.current_data.get_zs_mode();
	feu.N = feuIDToN[feu.current_data.get_Feu_ID()];
	if(feu.N<0) return true;//do not read this feu
      } else if (feu.FeuHeaderLine==1) {
	feu.EventID = feu.current_data.get_data();
      } else if (feu.FeuHeaderLine==2) {
	feu.TimeStamp = feu.current_data.get_data();
      } else if (feu.FeuHeaderLine==3) {

	feu.FineTimeStamp = feu.current_data.get_finetstp();//here
	if (FineTimeStamp[feu.N]<0) {//first !
	  FineTimeStamp[feu.N]=feu.FineTimeStamp;
	  //cout << feu.EventID << " FeuN =" << feu.N << " feu.TimeStamp=" << feu.TimeStamp  <<  " fine_time_stamp = " << feu.FineTimeStamp << endl;
	}
	
	feu.isample_prev = feu.isample;
	feu.isample = feu.current_data.get_sample_ID();
        feu.FeuHeaderLoaded = true;
	if (feu.isample!=feu.isample_prev+1) {
	  cerr << "problem in sample index in ReadFeuHeaders for feu N "<<feu.N 
	       << "("<<feu.isample<<"!="<<feu.isample_prev+1<<")"<< endl;
          bad_event = true;
	  // 	  break;
	}
      }
      feu.data_to_treat = false;
      feu.FeuHeaderLine++;
    } else if (feu.FeuHeaderLine>8 && feu.current_data.is_Feu_header()) {
      cerr << "problem in Feu header in feu "<<feu.N<<", FeuHeaderLine "<<feu.FeuHeaderLine<< endl;
      bad_event = true;
      break;
    } else if (feu.FeuHeaderLine>3 && !feu.current_data.is_Feu_header()) break;  // header finished
    badreadfg = feu.dfile->ReadBuffer((char*)&feu.current_data, sizeof(feu.current_data));
    if (badreadfg) return true;  // failed
    feu.current_data.ntohs_(); feu.data_to_treat = true;
  }
  return false;
}



//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

bool My_LorentzAngle::ReadDreamData(FeuReadOut& feu, bool skipping) {
  register bool badreadfg = false, got_channel_id = false;
  register int ichannel=0;

  if (!feu.FeuHeaderLoaded) {
    cerr<<"Error in ReadDreamData, Feu header not loaded"<<endl;
    return true;
  }

  if (!feu.data_to_treat) {
    badreadfg = feu.dfile->ReadBuffer((char*)&feu.current_data, sizeof(feu.current_data));
    if (badreadfg) return true;  // failed
    feu.current_data.ntohs_(); feu.data_to_treat = true;
  }

  while (true) {

    if (feu.FeuHeaderLine>3 && !feu.current_data.is_Feu_header()) {
      if (feu.DataHeaderLine<4 && feu.current_data.is_data_header()) {
        feu.asicN = feu.current_data.get_dream_ID();
        feu.detN = feu.N*8 + feu.asicN;
//         detN = det_n_by_asic[det];
        feu.DataHeaderLine++;
        feu.data_to_treat = false;
      } else if (feu.DataHeaderLine>3 && feu.current_data.is_data_header()) {
        bad_event = true;
        cerr << "problem in data header in ReadDreamData, DataHeaderLine "<<feu.DataHeaderLine<<" feuN "<<feu.N<<endl;
        return true;
      } else if (feu.DataHeaderLine>0) {
	if (feu.current_data.is_data() && !feu.zs_mode) {
	  // 	  channelN = mapping(det_type_by_asic[det], ichannel);
	  feu.channelN = ichannel;
	  if (!skipping && !bad_event && feu.detN<Ndet && feu.channelN>-1 && feu.channelN<NstripMax && feu.isample<Nsample)
            StripAmpl[feu.detN][feu.channelN][feu.isample] = feu.current_data.get_data();
	  ichannel++;
          feu.data_to_treat = false;
	} else if (feu.current_data.is_data_zs() && feu.zs_mode) {
	  if (!got_channel_id) {
	    ichannel = feu.current_data.get_channel_ID();
	    // 	    channelN = mapping(det_type_by_asic[asicN],ichannel);
	    feu.channelN = ichannel;
	    got_channel_id = true;
	  } else {
            if (!skipping && !bad_event && feu.detN<Ndet && feu.channelN>-1 && feu.channelN<NstripMax && feu.isample<Nsample)
              StripAmpl[feu.detN][feu.channelN][feu.isample] = feu.current_data.get_data();
	    got_channel_id = false;
	  }
          feu.data_to_treat = false;
	} else if (feu.current_data.is_data_trailer()) {
          if (ichannel!=64 && !feu.zs_mode) {
            bad_event = true;
	    cerr << "problem in channel number in non-ZS mode, ichannel " <<ichannel<<" feuN "<<feu.N<< endl;
	    return true;
	  }
	  if (got_channel_id) {
            bad_event = true;
	    cerr << "problem in ZS data, got_channel_id true"<<" feuN "<<feu.N << endl;
	    return true;
	  }
	  ichannel=0; feu.asicN=0; feu.detN=0; feu.channelN=0; feu.DataHeaderLine=0;
          feu.data_to_treat = false;
	}
      } else if (feu.current_data.is_final_trailer()) break;  // Dream data finished
    }

    badreadfg = feu.dfile->ReadBuffer((char*)&feu.current_data, sizeof(feu.current_data));
    if (badreadfg) return true;  // failed
    feu.current_data.ntohs_(); feu.data_to_treat = true;
  }
  return false;
}


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

bool My_LorentzAngle::ReadFeuTrailer(FeuReadOut& feu, bool skipping) {
  register bool badreadfg;


  if (!feu.data_to_treat) {
    badreadfg = feu.dfile->ReadBuffer((char*)&feu.current_data, sizeof(feu.current_data));
    if (badreadfg) return true;  // failed
    feu.current_data.ntohs_(); feu.data_to_treat = true;
  }

  while (true) {

    if (feu.current_data.is_final_trailer()) {
      if (feu.channelN != 0) {
        bad_event = true;
	cerr << "problem in channel number, feu.channelN "<<feu.channelN<<" feuN "<<feu.N << endl;
	return true;
      }
//       if (got_channel_id) { 
// 	cout << "problem in ZS data at final trailer, got_channel_id true"<<" feuN "<<feu.N << endl;
// 	return true;
//       }
      if (feu.isample == (Nsample-1)) {
	feu.isample=-1; feu.isample_prev=-2;
        feu.event_completed = true;
//           treer->Fill();
//           My_CleanStripAmpl();
      }
// 	isample_nb++;
      feu.N = 0;
      feu.FeuHeaderLine = 0;
      feu.FeuHeaderLoaded = false;
      feu.zs_mode = false;
      feu.data_to_treat = false;
      feu.dfile->ReadBuffer((char*)&feu.current_data, sizeof(feu.current_data));
      break;
    }

    badreadfg = feu.dfile->ReadBuffer((char*)&feu.current_data, sizeof(feu.current_data));
    if (badreadfg) return true;  // failed
    feu.current_data.ntohs_(); feu.data_to_treat = true;

  }
  return false;
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

bool My_LorentzAngle::SkipEvent(FeuReadOut& feu) {
  register bool badreadfg = false;

  while (!feu.event_completed) {

    badreadfg = ReadFeuHeaders(feu);  // read feu header if not done
    if (badreadfg) return true;  // failed

    badreadfg = ReadDreamData(feu, true);  // read data in skipping mode (true = no fill up of StripAmpl)
    if (badreadfg) return true;  // failed

    badreadfg = ReadFeuTrailer(feu);  // read feu trailer
    if (badreadfg) return true;  // failed

  }

  feu.isample=-1; feu.isample_prev=-2;
  feu.NewEvent();
  return false;
}


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

bool My_LorentzAngle::ReadEvent(FeuReadOut& feu) {
  register bool badreadfg = false;

  while (!feu.event_completed) {

    badreadfg = ReadFeuHeaders(feu);  // read feu header if not done
    if (badreadfg) return true;  // failed

    badreadfg = ReadDreamData(feu, false);  // read data in skipping mode (false = fill up StripAmpl)
    if (badreadfg) return true;  // failed

    badreadfg = ReadFeuTrailer(feu);  // read feu trailer
    if (badreadfg) return true;  // failed

  }

  feu.isample=-1; feu.isample_prev=-2;
  feu.NewEvent();
  return false;
}


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void My_LorentzAngle::CleanStripAmpl() {
    for (register int idet=0; idet<Ndet; idet++)
      for (register int istrip=0; istrip<NstripMax; istrip++)
        for (register int itime=0; itime<Nsample; itime++) StripAmpl[idet][istrip][itime]=0;
}


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

My_LorentzAngle::~My_LorentzAngle() {
}


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

void signal_handler(int sig) {
  stopPgm = true;
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

extern void InitGui();
VoidFuncPtr_t initfuncs[] = { InitGui, 0};

TROOT root("Analyse", " TView Analyse ", initfuncs);

int main (int argc, char **argv)  {
  cerr<<"argc "<<argc<<endl;
  dataFileNames.clear();
  if (argc>1) {
//     dataFileName = argv[1];
    cout<<"Data file names: "<<endl;
    for (register int ii=1; ii<argc; ii++) {
      dataFileNames.push_back(string(argv[ii]));
      cout<<" - "<<dataFileNames.back()<<endl;
    }
//     sprintf(dataFileName,"%s",argv[1]);
  } else {
    cerr<<"No data file name given, exiting..."<<endl;
    exit(1);
  }

  sigactif.sa_handler = signal_handler;
//   sigactif.sa_mask = 0;
  sigactif.sa_flags = 0;
  sigaction(SIGINT, &sigactif, 0);

  TApplication theApp("App", &argc, argv);

  time(&td);
  dcp=localtime(&td);
  cout<<endl;
  printf("On est le %02d/%02d/%04d, il est %02d:%02d:%02d      ",
	 dcp->tm_mon+1,    //mois
	 dcp->tm_mday,//jour
	 dcp->tm_year+1900,//annee
	 dcp->tm_hour,//heure
	 dcp->tm_min,//minute
	 dcp->tm_sec);//seconde

  cout<<endl;
  cout<<endl;
  cout<<"             ####################################"<<endl;
  cout<<"             | Welcome to the Dream data reader |"<<endl;
  cout<<"             ####################################"<<endl;

  My_LorentzAngle *process = new My_LorentzAngle();
  process->ReadFeuIDs("FeuIDs.txt"),
  process->ReadOnline();

  time(&td);
  dcp=localtime(&td);
  cout<<endl;
  printf("il est %02d:%02d:%02d      ",
	 dcp->tm_hour,//heur
	 dcp->tm_min,//minute
	 dcp->tm_sec);//seconde0

//   theApp.Run();
  return 0;
}
