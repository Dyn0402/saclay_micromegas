ObjSuf        = o
SrcSuf        = cxx
ExeSuf        =
DllSuf        = so

OutPutOpt     = -o

MYINCLUDE     = ~/.
MYLIB         = $(PPATH)/lib
OUTPUTDIR     = ./
ROOTCFLAGS    = $(shell root-config --cflags)
ROOTLIBS      = $(shell root-config --libs) -lMinuit
ROOTGLIBS     = $(shell root-config --glibs) -lMinuit



# Linux with egcs
CXX           = g++
CXXFLAGS      = -g -O -Wall -Wno-deprecated -fexceptions -fPIC $(ROOTCFLAGS) -I$(MYINCLUDE) 
LD            = g++
LIBS          = $(ROOTLIBS) -lNetx -lm -ldl -rdynamic
# GLIBS         = $(ROOTGLIBS) -L/usr/local/root/xrootd/xrootd-3.2.0/lib64 -Wl,-rpath,/usr/local/root/xrootd/xrootd-3.2.0/lib64  -lXrdUtils -lXrdClient -L/usr/lib64 -L$(MYLIB) -lXpm -lX11 -lm -ldl -rdynamic -lpthread
# GLIBS         = $(ROOTGLIBS) -L/afs/in2p3.fr/system/amd64_sl5/usr/local/root/xrootd/xrootd-3.2.0/lib64 -Wl,-rpath,/afs/in2p3.fr/system/amd64_sl5/usr/local/root/xrootd/xrootd-3.2.0/lib64  -lXrdUtils -lXrdClient -L/usr/lib64 -L$(MYLIB) -lXpm -lX11 -lm -ldl -rdynamic -lpthread
GLIBS         = $(ROOTGLIBS) -L/usr/local/products/xrootd/root/3.3.2/lib64 -Wl,-rpath,/usr/local/products/xrootd/root/3.3.2/lib64   -L/usr/lib64 -L$(MYLIB) -lXpm -lX11 -lm -ldl -rdynamic -lpthread
LDFLAGS       =  $(GLIBS) 
SOFLAGS       = -shared

#------------------------------------------------------------------------------

all:    DreamDataReader

lib:    libEvents.so


clean:
	@rm -f $(OBJS) *Dict.* *dict* core *.o StartAnalysis T2KDataReader DreamDataReader StagingFile Staging

.SUFFIXES: .$(SrcSuf)

.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $< 

.o:
	$(LD) $< -o $(OUTPUTDIR)$* $(LDFLAGS)


StartAnalysis:  StartAnalysis.o Events.o MyMappage.o PixelMMDecoding.o MyDict.o
	$(LD) $^ -o StartAnalysis $(LDFLAGS) -lMatrix

MyDict.cxx: Events.h Linkdef.h
	rootcint -f $@ -c $(CXXFLAGS) -p $^

libEvents.so: Events.o MyDict.o
	$(CXX) -shared -o $@ $< MyDict.o







