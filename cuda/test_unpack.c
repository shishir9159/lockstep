#include <stdio.h>
#include "fp4_unpack.cuh"
static float e4m3_to_float(uint8_t b){
    int s=(b>>7)&1, e=(b>>3)&0xF, m=b&7; float v;
    if(e==0) v=(float)m/8.0f*0.015625f;      /* 2^-6 */
    else { float p=1.0f; int k=e-7; if(k>=0) for(int i=0;i<k;i++)p*=2; else for(int i=0;i<-k;i++)p/=2;
           v=(1.0f+(float)m/8.0f)*p; }
    return s?-v:v;
}
int main(void){
    int bad=0;
    /* exhaustive over all 16^4 nibble quadruples */
    for(int n3=0;n3<16;n3++)for(int n2=0;n2<16;n2++)for(int n1=0;n1<16;n1++)for(int n0=0;n0<16;n0++){
        uint32_t p=(uint32_t)n0|((uint32_t)n1<<4)|((uint32_t)n2<<8)|((uint32_t)n3<<12);
        uint32_t got=fp4x4_to_e4m3x4(p);
        int nib[4]={n0,n1,n2,n3};
        for(int i=0;i<4;i++){
            uint8_t g=(uint8_t)(got>>(8*i)), w=fp4_code_to_e4m3_ref((uint8_t)nib[i]);
            if(g!=w){ if(bad<5)printf("MISMATCH code=%x got=%02x want=%02x\n",nib[i],g,w); bad++; }
            float fv=e4m3_to_float(g), rv=fp4_code_to_float((uint8_t)nib[i]);
            if(fv!=rv && !(fv==0&&rv==0)){ if(bad<5)printf("VALUE code=%x e4m3=%f e2m1=%f\n",nib[i],fv,rv); bad++; }
        }
    }
    printf("exhaustive 16^4 quadruples: %d mismatches\n", bad);
    /* 8-at-a-time path */
    uint32_t lo,hi; fp4x8_to_e4m3x8(0xFEDCBA98u,&lo,&hi);
    printf("fp4x8(0xFEDCBA98) -> %08x %08x  (codes 8,9,a,b | c,d,e,f)\n", lo, hi);
    for(int c=0;c<16;c++) printf("  code %2d -> 0x%02X = %g\n", c, fp4_code_to_e4m3_ref(c), fp4_code_to_float(c));
    return bad!=0;
}
