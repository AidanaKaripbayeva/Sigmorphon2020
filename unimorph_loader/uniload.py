
import pandas as _pn

def read_unimorph_tsv(filename):
	return _pn.read_csv(filename,sep="\t",names=["lemma","form","features"],dtype=str)

MAX_LANGSPEC = 10

#taken from https://unimorph.github.io/doc/unimorph-schema.pdf
unimorph_dimension_names = ["Aktionsart",
"Animacy",
"Argument Marking",
"Aspect",
"Case",
"Comparison",
"Definiteness",
"Deixis",
"Evidentiality",
"Finiteness",
"Gender",
"Information Structure",
"Interrogativity",
"Language-Specific Features",
"Mood",
"Number",
"Part of Speech",
"Person",
"Polarity",
"Politeness",
"Possession",
"Switch-Reference",
"Tense",
"Valency",
"Voice"]

unimorph_labels = {
"Aktionsart":["ACCMP","ACH","ACTY","ATEL","DUR","DYN","PCT","SEMEL","STAT","TEL"]
,"Animacy":["ANIM","HUM","INAN","NHUM"]
,"Argument Marking":["ARGAC3S"]
,"Aspect":["HAB","IPFV","ITER","PFV","PRF","PROG","PROSP"]
,"Case":["ABL","ABS","ACC","ALL","ANTE","APPRX","APUD","AT","AVR",
		"BEN","BYWAY","CIRC","COM","COMPV","DAT","EQTV","ERG","ESS",
		"FRML","GEN","IN","INS","INTER","NOM","NOMS","ON","ONHR","ONVR",
		"POST","PRIV","PROL","PROPR","PROX","PRP","PRT",
		"REL","REM","SUB","TERM","TRANS","VERS","VOC"]
,"Comparison":["AB","CMPR","EQT","RL","SPRL"]
,"Definiteness":["DEF","INDF","NSPEC","SPEC"]
,"Deixis":["ABV","BEL","EVEN","MED","NOREF","NVIS","PHOR","PROX","REF1","REF2","REMT","VIS"]
,"Evidentiality":["ASSUM","AUD","DRCT","FH","HRSY","INFER","NFH","NVSEN","QUOT","RPRT","SEN"]
,"Finiteness":["FIN","NFIN"]
,"Gender":["BANTU"+str(i) for i in range(1,24)] + ["FEM", "MASC"] + ["NAKH"+str(i) for i in range(1,9)] + ["NEUT"]
,"Information Structure":["FOC","TOP"]
,"Interrogativity":["DECL","INT"]
,"Language-Specific Features": ["LGSPEC"+str(i) for i in range(MAX_LANGSPEC)]
,"Mood":["ADM","AUNPRP","AUPRP","COND","DEB","DED","IMP","IND","INTEN","IRR","LKLY","OBLIG","OPT","PERM","POT","PURP","REAL","SBJV","SIM"]
,"Number":["DU","GPAUC","GRPL","INVN","PAUC","PL","SG"]
,"Part of Speech":["ADJ","ADP","ADV","ART","AUX","CLF","COMP","CONJ",
					"DET","INTJ","N","NUM","PART","PRO","PROPN",
					"V","V.CVB","V.MSDR","V.PTCP"]
,"Person":["0","1","2","3","4","EXCL","INCL","OBV","PRX"]
,"Polarity":["POS","NEG"]
,"Politeness":["AVOID","COL","ELEV","FOREG","FORM","HIGH","HUMB","INFM","LIT","LOW","POL","STELEV","STSUPR"]
,"Possession":["ALN","NALN",
				"PSS1D","PSS1DE","PSS1DI","PPS1P","PSS1PE","PSS1PI","PSS1S",
				"PSS2D","PSS2DF","PSS2DM","PSS2P","PSS2PF","PSS2PM",
				"PSS2S","PSS2SF","PSS2SFORM","PSS2SINFM","PSS2SM",
				"PSS3D","PSS3DF","PSS3DM","PSS3P","PSS3PF","PSS3PM",
				"PSS3S","PSS3SF","PSS3SM",
				"PSSD"]
,"Switch-Reference":["CN_R_MN","DS","DSADV","LOG","OR","SEQMA","SIMMA","SS","SSADV"]
,"Tense":["1DAY","FUT","HOD","IMMED","PRS","PST","RCT","RMT"]
,"Valency":["APPL","CAUS","DITR","IMPRS","INTR","RECP","REFL","TR"]
,"Voice":["ACFOC","ACT","AGFOC","ANTIP","BFOC","CFOC","DIR","IFOC","INV","LFOC","MID","PASS","PFOC"]
}
