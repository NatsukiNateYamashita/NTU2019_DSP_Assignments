#include <string.h>
#include <vector>
#include <map>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include "Ngram.h"
#define F_NAME_NUM 128

using namespace std;
//usage ./mydisambig -text testdata/xx.txt -map ZhuYin-Big5.map -lm bigram.lm -order 2 > $output

int main( int argc, char *argv[]){
		int argv_tmp = 1;
		char f_text[F_NAME_NUM];
		char f_map[F_NAME_NUM];
		char f_lm[F_NAME_NUM];
		char f_result[F_NAME_NUM];

		strcpy( f_text, argv[1]);
		strcpy( f_map, argv[2]);
		strcpy( f_lm, argv[3]);
		strcpy( f_result, argv[4]);

    int ngram_order = 2;
		Vocab voc;
		Ngram lm( voc, ngram_order);
		{
				File lmFile( f_lm, "r");
				lm.read( lmFile);
				lmFile.close( );
		}
/////////////////
/////GET_MAP/////
/////////////////
		map<string, vector<string>> z_map;
		char line[1000000];
		FILE* map_fp = fopen( f_map, "r");
		while( fgets( line, sizeof( line), map_fp)!=NULL){
				char key_col[3];
				strncpy( key_col, line, 2);
				string key( key_col);
				vector<string> voc_tmp;
				for( int i = 2; i < strlen( line); i++){
						if( line[i]==' ' || line[i] == '\t'){
								char rest[3];
								strncpy( rest, line+i+1, 2);
								string tmp( rest);
								voc_tmp.push_back( tmp);
						}
						z_map[key]=voc_tmp;
				}
		}
/////////////////
/////GET_TEXT////
/////////////////
		FILE* fp = fopen( f_text, "r");
		if( fp == NULL){
				printf( "No such a file\n");
				return 1;
		}
		ofstream outputfile(f_result);
		while( fgets( line, sizeof( line), fp)!=NULL){
				vector<string> voc_tmp;
				for( int i = 0; i < strlen( line); i++){
						if( line[i]!=' ' && line[i] !='\n'){
								char rest[3];
								strncpy( rest, line+i, 2);
								string tmp( rest);
								voc_tmp.push_back( tmp);
								i++;
						}
				}

				string start( "<s>");
				VocabIndex st = voc.getIndex( start.c_str( ));
				string end( "</s>");
				VocabIndex ed = voc.getIndex( end.c_str( ));
				int cnt = voc_tmp.size( );
				vector<double> fw[cnt];
				vector<int> bw[cnt];
				vector<string> state[cnt];
				vector<int> index[cnt];
				state[0] = z_map[voc_tmp[0]];

				for( int i = 0; i < state[0].size( ); i++){
						VocabIndex wid = voc.getIndex( state[0][i].c_str( ));
						if( wid == Vocab_None){
								wid = voc.getIndex( Vocab_Unknown);
						}
						index[0].push_back( wid);
						VocabIndex context[] = {st,Vocab_None};
						fw[0].push_back( lm.wordProb( wid, context));
				}
				for( int i = 1; i < cnt; i++){
						state[i] = z_map[voc_tmp[i]];
						for( int j = 0; j < state[i].size( ); j++){
								VocabIndex wid = voc.getIndex( state[i][j].c_str( ));
								if( wid == Vocab_None){
										wid = voc.getIndex( Vocab_Unknown);
								}
								index[i].push_back( wid);
								fw[i].push_back( -99999999999);
								bw[i].push_back( 0);
								for( int k = 0; k < state[i-1].size( ); k++){
										VocabIndex context[] = {index[i-1][k], Vocab_None};
										if( ( lm.wordProb( wid, context)+ fw[i-1][k]) > fw[i][j]){
												fw[i][j] = lm.wordProb( wid, context) + fw[i-1][k];
												bw[i][j] = k;
										}
								}
						}
				}
				/////////////////
				/////CALC_PROB///
				/////////////////
				vector<string> output;
				double max = -99999999999;
				int max_index = 0;
				for( int i = 0; i < fw[cnt-1].size( ); i++){
						VocabIndex context[] = {index[cnt-1][i], Vocab_None};
						double max_tmp = lm.wordProb( ed, context) + fw[cnt-1][i];
						if( max_tmp> max){
								max_index = i;
								max = max_tmp;
						}
				}
				output.push_back( state[cnt-1][max_index]);
				for( int i = cnt-1; i >= 1; i--){
						output.push_back( state[i-1][bw[i][max_index]]);
						max_index = bw[i][max_index];
				}
				/////////////////
				/////OUTPUT//////
				/////////////////
				outputfile<<"<s>";
				for( int i = cnt-1; i > -1 ; i--){
						outputfile<<" "<<output[i];
				}
				outputfile<<" </s>\n";
		}
		outputfile.close();
		return 0;
}
