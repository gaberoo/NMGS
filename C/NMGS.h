#ifndef NMGS_H
#define NMGS_H

#define DELIM ",\n"
#define MAX_LINE_LENGTH   10000
#define MAX_WORD_LENGTH   128

#define MIN_DBL -1.0e30
#define INIT_SIZE 100
#define TRUE  1
#define FALSE 0

#define OPTION  0      /* optional */
#define ALWAYS  1      /* required */

#define OUT_FILE_STUB    "-out"
#define INPUT_FILE       "-in"
#define SEED       	     "-l"
#define VERBOSE          "-v"
#define N_ITERATIONS     "-t"
#define RAREFY           "-r"
#define SAMPLE           "-s"
#define EXTRAPOLATE      "-e"
#define OUTPUT_SAMPLE    "-o"
#define N_BURN_ITER      "-b"
#define SAMPLE_ONLY      "-S"

#define CSV_FILE_SUFFIX         ".csv"
#define FREQ_FILE_SUFFIX        "_f.csv"
#define SAMPLE_FILE_SUFFIX      "_s.csv"
#define EXTRAPOLATE_FILE_SUFFIX "_e.csv"
#define META_FILE_SUFFIX        "_m.csv"

#define SAMPLE_DIR              "_sample"

#define ETA   0.01         
#define NU    0.01         
#define ALPHA 0.01       
#define BETA  0.01

#define THETA_INIT 10.0
#define I_INIT     10.0

#define DEF_MAX_ITER  50000
#define DEF_BURN_ITER 25000
#define N_SAMPLE      10

typedef struct s_Params
{	
  int nMaxIter;        /* no. of iterations */
  int nBurnIter;       /* no. of burn-in iterations */
  int nL;              /* seed */
  char *szInputFile;   /* csv input file */
  char *szOutFileStub; /* output file stub */
  int bSample;         /* resample for fitting purposes */
  int nExtrapolate;    /* extrapolate to new sample size */
  int bOutputSample;   /* output samples */
  int nRarefy;         /* rarefy sample to lowest sample size */
  int bSampleOnly;     /* skip Gibbs sampling step */
} t_Params;

typedef struct s_Data
{
  int nN;
  int nS;
  int nSize;
  int **aanX;
  char **aszSampleNames;
  char **aszOTUNames;
} t_Data;

/* User defined functions */

void getCommandLineParams(t_Params *ptParams,int argc,char *argv[]);

void readAbundanceData(const char *szFile, t_Data *ptData);

void Stirling(double ***paadStirlingMatrix,double** padNormMatrix,unsigned long n);

int maxX(t_Data *ptData);

void sumColumns(int *anC, int **aanX, int nN, int nS);

void sumRows(int *anR, int **aanX, int nN, int nS);

int Sum(int *anT, int nN);

double safeexp(double x);

void generateData(gsl_rng* ptGSLRNG, t_Data *ptData, int nN, double dTheta, int *anJ, double* adI, int nS, double *adM);

double calcNLLikelihood(t_Data *ptData, double dTheta, double* adI, double *adM);

void addUnobserved(t_Data* ptDataR, t_Data *ptData);

void generateDataStick(gsl_rng* ptGSLRNG, t_Data *ptData, int nN, double dTheta, int *anJ, double* adI, int nS, double *adM, int *pnSDash, double **padMDash);

double calcMeanSpeciesEntropy(t_Data *ptData);

int calcR(t_Data *ptData);

int selectIntCat(gsl_rng* ptGSLRNG, int nN, int* anN);

void generateDataHDP(gsl_rng* ptGSLRNG, t_Data *ptData, int nN, double dTheta, int *anJ, double* adI, int **panT);

void generateDataFixed(gsl_rng* ptGSLRNG, t_Data *ptData, int nN, double dTheta, int *anJ, double* adI, int nS, double *adM);

void writeAbundanceData(const char *szFile, t_Data *ptData);

double sampleTheta(gsl_rng *ptGSLRNG, double dTheta, int* anTR, int nN, int nS);

void sampleMetacommunity(int nIter, gsl_rng *ptGSLRNG, double *adM, double **aadMStore, double dTheta, int nS, int *anTC);

void sampleMetacommunityHDP(gsl_rng *ptGSLRNG, double *adM, double dTheta, int nS, int *anT);

void sampleImmigrationRates(int nIter, gsl_rng *ptGSLRNG, double **aadIStore, int nN, double *adW, double *adS, double* adJ, int* anTR);

void sampleT(int nIter, gsl_rng* ptGSLRNG, int** aanX, int **aanT, double** aadIStore, double** aadMStore, 
	     double** aadStirlingMatrix, int nN, int nS, double *adLogProbV, double *adProbV, double *adCProbV);

void outputSamples(char *szOutputDir,int nIter, int nMaxIter, gsl_rng* ptGSLRNG, int nN, int nS, t_Params *ptParams, t_Data *ptData,
		   double *adThetaStore, int *anJ, double **aadIStore, double** aadMStore);

void extrapolateSamples(int nIter, int nMaxIter, gsl_rng* ptGSLRNG, int nN, int nS, t_Params *ptParams, t_Data *ptData,
			double *adThetaStore, int *anJ, double **aadIStore, double** aadMStore);

void extrapolateSamples2(char *szOutputDir, int nIter, int nMaxIter, gsl_rng* ptGSLRNG, int nE, int nN, int nS, t_Params *ptParams, t_Data *ptData,
			 double *adThetaStore, int *anJ, double **aadIStore, int** aanTStore);

int minJ(t_Data *ptData);

void rarefy(gsl_rng *ptGSLRNG,int nMaxJ, t_Data *ptDataR, t_Data *ptData);

void extrapolateDataHDP(gsl_rng* ptGSLRNG, t_Data *ptData, int nN, int nE, double dTheta, double* adI, int* anTSample);

void copyData(t_Data* ptDataR, t_Data *ptData);

void writeTheta(const t_Params *ptParams, int nMaxIter, int nN, 
                const double* adThetaStore, double** aadIStore);
void readTheta(const t_Params *ptParams, int nMaxIter, int nN, 
               double* adThetaStore, double** aadIStore);

#endif
