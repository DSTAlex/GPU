#include <iostream>
#include <vector>

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}


// sequential exclusive scan on the CPU
std::vector<int> scan_cpu(const std::vector<int>& x)
{
    std::vector<int> res = std::vector<int>(x.size());
    res[0] = 0;
    // for (int i = 1; i < x.size() - 1; i++)
    // {
    //     res[i] = res[i-1] + x[i];
    // }
    return res;
}


namespace kernel {
    
// CUDA kernel of the parallel exclusive scan on the GPU (naive version)
template<int T>
__global__
void scan_gpu1(int* x)
{
    // ...
}

} // namespace kernel


// parallel exclusive scan on the GPU (naive version)
template<int T>
std::vector<int> scan_gpu1(const std::vector<int>& x)
{
    // ...
    return {};
}


namespace kernel {

// CUDA kernel of the parallel exclusive scan on the GPU (optimized version)
template<int T>
__global__
void scan_gpu2(int* x)
{
    // ...
}

} // namespace kernel


// parallel exclusive scan on the GPU (optimized version)
template<int T>
std::vector<int> scan_gpu2(const std::vector<int>& x)
{
    // ...
    return {};
}



int main()
{
    std::cout << "Testing scan_cpu() ________________" << std::endl;
    {
        std::cout << "Test 1 ";
        const std::vector<int> x = {3,2,5,6,8,7,4,1};
        const std::vector<int> y_sol = {0,3,5,10,16,24,31,35};
        const std::vector<int> y_test = scan_cpu(x);
        if(y_test != y_sol) {
            std::cout << "failure" << std::endl;
            std::cout << "  expected = [0,3,5,10,16,24,31,35]" << std::endl;
            std::cout << "  get      = [";
            for(int val : y_test) std::cout << val << ",";
            std::cout << "]" << std::endl;
        } else {
            std::cout << "success" << std::endl;
        }
    }
    {
        std::cout << "Test 2 ";
        const std::vector<int> x = {16,15,4,14,10,1,13,12,2,11,9,7,8,5,3,6};
        const std::vector<int> y_sol = {0,16,31,35,49,59,60,73,85,87,98,107,114,122,127,130};
        const std::vector<int> y_test = scan_cpu(x);
        if(y_test != y_sol) {
            std::cout << "failure" << std::endl;
        } else {
            std::cout << "success" << std::endl;
        }
    }
    {
        std::cout << "Test 3 ";
        const std::vector<int> x = {632,680,365,320,488,724,484,955,252,271,76,672,679,321,368,295,857,690,507,531,2,28,532,32,514,547,963,852,381,313,332,944,19,919,810,866,519,1016,296,404,279,215,614,863,123,318,562,15,16,143,913,86,197,387,248,357,193,996,142,559,440,433,972,485,262,46,838,760,795,743,635,56,1001,968,410,639,109,407,463,952,744,787,78,352,581,414,586,630,833,993,957,419,584,763,974,284,98,331,317,468,1014,990,104,379,231,748,396,3,436,667,652,426,176,776,304,656,842,577,201,906,829,448,890,751,583,203,747,730,256,716,300,821,861,722,814,254,605,774,421,66,650,801,699,647,615,311,265,437,309,61,316,425,641,846,626,512,960,770,878,495,849,527,453,250,9,560,860,102,268,501,435,794,253,454,726,643,444,55,35,240,31,791,796,649,688,479,870,826,905,523,958,912,725,956,480,521,684,892,831,903,191,280,534,836,146,859,478,773,628,278,813,671,708,145,879,766,373,400,666,216,694,703,754,343,403,441,385,155,205,683,6,589,732,997,759,417,689,151,232,978,627,588,492,554,604,273,546,458,241,777,929,114,761,38,719,696,943,159,41,359,486,353,568,921,237,633,502,644,799,558,362,496,63,80,194,409,182,360,610,10,845,835,494,733,263,7,267,971,39,995,762,299,272,141,651,195,753,659,809,466,742,616,51,163,530,312,413,876,579,837,173,940,367,818,758,983,805,40,593,825,157,223,677,211,307,784,1013,737,168,966,789,383,11,106,69,91,475,487,259,597,973,305,936,549,989,975,209,21,621,953,423,827,286,662,72,50,180,856,363,117,294,77,49,398,264,850,811,950,333,65,839,53,364,493,125,686,459,472,1004,524,303,119,13,158,338,128,47,236,569,242,619,490,959,1011,893,908,326,947,823,351,636,18,1020,187,452,339,207,483,920,346,314,243,482,171,188,949,134,302,768,642,319,79,819,347,397,199,390,386,969,467,965,620,824,550,769,22,681,755,600,251,244,602,36,82,160,718,334,881,497,707,110,631,380,704,565,93,504,522,999,606,391,609,71,473,212,172,166,239,189,1012,665,655,345,572,930,506,682,874,156,525,603,793,126,457,67,645,234,225,269,162,625,539,712,179,734,416,101,570,657,230,335,942,739,802,914,970,221,392,1003,428,310,815,932,528,246,670,786,105,1023,52,355,461,23,375,219,536,904,954,206,709,498,517,376,186,405,901,402,935,692,442,133,174,429,598,1021,797,1008,97,623,500,75,90,790,601,1,393,477,840,520,108,745,668,224,328,103,741,476,214,136,247,1024,1010,653,948,366,855,27,678,418,673,591,4,135,356,923,808,880,822,832,226,864,370,804,340,420,887,1002,455,64,1009,361,238,113,931,533,578,127,260,816,746,138,8,503,875,738,896,408,729,354,154,567,543,344,934,613,721,946,57,401,573,474,469,687,705,505,675,685,982,301,800,701,571,164,637,337,60,865,984,491,828,450,900,438,564,980,937,220,714,587,283,349,723,5,1017,464,922,693,926,518,654,574,648,834,941,535,951,277,285,471,427,282,121,594,315,14,924,456,585,1007,161,88,713,752,782,196,192,629,537,927,165,107,17,576,976,749,541,330,481,499,916,406,781,94,434,84,888,820,711,658,415,663,552,767,765,961,788,323,557,358,70,515,510,341,222,979,147,290,30,115,509,853,538,177,169,348,377,988,1022,898,812,190,137,447,991,255,928,291,992,891,233,646,34,89,45,342,62,1005,200,430,785,395,806,731,298,74,394,548,580,542,899,877,884,664,218,540,867,872,624,669,868,422,695,139,213,617,886,925,1006,728,1018,702,727,967,717,235,772,68,25,382,217,736,130,350,112,167,964,208,439,553,275,894,592,858,843,526,288,470,327,122,764,508,229,674,596,446,118,465,26,111,150,153,529,697,844,622,462,599,140,750,261,895,830,324,945,249,451,411,841,608,20,329,297,270,227,95,516,885,638,281,287,590,981,445,185,372,918,175,276,807,986,183,33,81,308,771,12,873,44,87,779,37,706,59,42,783,412,563,210,149,740,595,124,735,575,99,378,131,700,24,915,556,371,907,513,384,202,369,555,611,388,862,817,293,691,882,257,607,198,96,1015,994,170,582,306,561,854,374,910,720,792,228,661,144,847,181,869,132,1000,48,775,325,204,85,939,634,909,566,612,399,897,92,83,431,780,120,871,292,58,938,322,258,889,998,443,43,618,389,336,274,757,489,551,902,54,933,987,129,803,29,184,883,460,511,676,152,245,715,544,545,640,698,266,962,289,424,911,148,660,798,1019,178,985,100,851,449,432,848,710,756,977,116,917,778,73};
        const std::vector<int> y_sol = {0,632,1312,1677,1997,2485,3209,3693,4648,4900,5171,5247,5919,6598,6919,7287,7582,8439,9129,9636,10167,10169,10197,10729,10761,11275,11822,12785,13637,14018,14331,14663,15607,15626,16545,17355,18221,18740,19756,20052,20456,20735,20950,21564,22427,22550,22868,23430,23445,23461,23604,24517,24603,24800,25187,25435,25792,25985,26981,27123,27682,28122,28555,29527,30012,30274,30320,31158,31918,32713,33456,34091,34147,35148,36116,36526,37165,37274,37681,38144,39096,39840,40627,40705,41057,41638,42052,42638,43268,44101,45094,46051,46470,47054,47817,48791,49075,49173,49504,49821,50289,51303,52293,52397,52776,53007,53755,54151,54154,54590,55257,55909,56335,56511,57287,57591,58247,59089,59666,59867,60773,61602,62050,62940,63691,64274,64477,65224,65954,66210,66926,67226,68047,68908,69630,70444,70698,71303,72077,72498,72564,73214,74015,74714,75361,75976,76287,76552,76989,77298,77359,77675,78100,78741,79587,80213,80725,81685,82455,83333,83828,84677,85204,85657,85907,85916,86476,87336,87438,87706,88207,88642,89436,89689,90143,90869,91512,91956,92011,92046,92286,92317,93108,93904,94553,95241,95720,96590,97416,98321,98844,99802,100714,101439,102395,102875,103396,104080,104972,105803,106706,106897,107177,107711,108547,108693,109552,110030,110803,111431,111709,112522,113193,113901,114046,114925,115691,116064,116464,117130,117346,118040,118743,119497,119840,120243,120684,121069,121224,121429,122112,122118,122707,123439,124436,125195,125612,126301,126452,126684,127662,128289,128877,129369,129923,130527,130800,131346,131804,132045,132822,133751,133865,134626,134664,135383,136079,137022,137181,137222,137581,138067,138420,138988,139909,140146,140779,141281,141925,142724,143282,143644,144140,144203,144283,144477,144886,145068,145428,146038,146048,146893,147728,148222,148955,149218,149225,149492,150463,150502,151497,152259,152558,152830,152971,153622,153817,154570,155229,156038,156504,157246,157862,157913,158076,158606,158918,159331,160207,160786,161623,161796,162736,163103,163921,164679,165662,166467,166507,167100,167925,168082,168305,168982,169193,169500,170284,171297,172034,172202,173168,173957,174340,174351,174457,174526,174617,175092,175579,175838,176435,177408,177713,178649,179198,180187,181162,181371,181392,182013,182966,183389,184216,184502,185164,185236,185286,185466,186322,186685,186802,187096,187173,187222,187620,187884,188734,189545,190495,190828,190893,191732,191785,192149,192642,192767,193453,193912,194384,195388,195912,196215,196334,196347,196505,196843,196971,197018,197254,197823,198065,198684,199174,200133,201144,202037,202945,203271,204218,205041,205392,206028,206046,207066,207253,207705,208044,208251,208734,209654,210000,210314,210557,211039,211210,211398,212347,212481,212783,213551,214193,214512,214591,215410,215757,216154,216353,216743,217129,218098,218565,219530,220150,220974,221524,222293,222315,222996,223751,224351,224602,224846,225448,225484,225566,225726,226444,226778,227659,228156,228863,228973,229604,229984,230688,231253,231346,231850,232372,233371,233977,234368,234977,235048,235521,235733,235905,236071,236310,236499,237511,238176,238831,239176,239748,240678,241184,241866,242740,242896,243421,244024,244817,244943,245400,245467,246112,246346,246571,246840,247002,247627,248166,248878,249057,249791,250207,250308,250878,251535,251765,252100,253042,253781,254583,255497,256467,256688,257080,258083,258511,258821,259636,260568,261096,261342,262012,262798,262903,263926,263978,264333,264794,264817,265192,265411,265947,266851,267805,268011,268720,269218,269735,270111,270297,270702,271603,272005,272940,273632,274074,274207,274381,274810,275408,276429,277226,278234,278331,278954,279454,279529,279619,280409,281010,281011,281404,281881,282721,283241,283349,284094,284762,284986,285314,285417,286158,286634,286848,286984,287231,288255,289265,289918,290866,291232,292087,292114,292792,293210,293883,294474,294478,294613,294969,295892,296700,297580,298402,299234,299460,300324,300694,301498,301838,302258,303145,304147,304602,304666,305675,306036,306274,306387,307318,307851,308429,308556,308816,309632,310378,310516,310524,311027,311902,312640,313536,313944,314673,315027,315181,315748,316291,316635,317569,318182,318903,319849,319906,320307,320880,321354,321823,322510,323215,323720,324395,325080,326062,326363,327163,327864,328435,328599,329236,329573,329633,330498,331482,331973,332801,333251,334151,334589,335153,336133,337070,337290,338004,338591,338874,339223,339946,339951,340968,341432,342354,343047,343973,344491,345145,345719,346367,347201,348142,348677,349628,349905,350190,350661,351088,351370,351491,352085,352400,352414,353338,353794,354379,355386,355547,355635,356348,357100,357882,358078,358270,358899,359436,360363,360528,360635,360652,361228,362204,362953,363494,363824,364305,364804,365720,366126,366907,367001,367435,367519,368407,369227,369938,370596,371011,371674,372226,372993,373758,374719,375507,375830,376387,376745,376815,377330,377840,378181,378403,379382,379529,379819,379849,379964,380473,381326,381864,382041,382210,382558,382935,383923,384945,385843,386655,386845,386982,387429,388420,388675,389603,389894,390886,391777,392010,392656,392690,392779,392824,393166,393228,394233,394433,394863,395648,396043,396849,397580,397878,397952,398346,398894,399474,400016,400915,401792,402676,403340,403558,404098,404965,405837,406461,407130,407998,408420,409115,409254,409467,410084,410970,411895,412901,413629,414647,415349,416076,417043,417760,417995,418767,418835,418860,419242,419459,420195,420325,420675,420787,420954,421918,422126,422565,423118,423393,424287,424879,425737,426580,427106,427394,427864,428191,428313,429077,429585,429814,430488,431084,431530,431648,432113,432139,432250,432400,432553,433082,433779,434623,435245,435707,436306,436446,437196,437457,438352,439182,439506,440451,440700,441151,441562,442403,443011,443031,443360,443657,443927,444154,444249,444765,445650,446288,446569,446856,447446,448427,448872,449057,449429,450347,450522,450798,451605,452591,452774,452807,452888,453196,453967,453979,454852,454896,454983,455762,455799,456505,456564,456606,457389,457801,458364,458574,458723,459463,460058,460182,460917,461492,461591,461969,462100,462800,462824,463739,464295,464666,465573,466086,466470,466672,467041,467596,468207,468595,469457,470274,470567,471258,472140,472397,473004,473202,473298,474313,475307,475477,476059,476365,476926,477780,478154,479064,479784,480576,480804,481465,481609,482456,482637,483506,483638,484638,484686,485461,485786,485990,486075,487014,487648,488557,489123,489735,490134,491031,491123,491206,491637,492417,492537,493408,493700,493758,494696,495018,495276,496165,497163,497606,497649,498267,498656,498992,499266,500023,500512,501063,501965,502019,502952,503939,504068,504871,504900,505084,505967,506427,506938,507614,507766,508011,508726,509270,509815,510455,511153,511419,512381,512670,513094,514005,514153,514813,515611,516630,516808,517793,517893,518744,519193,519625,520473,521183,521939,522916,523032,523949,524727};
        const std::vector<int> y_test = scan_cpu(x);
        if(y_test != y_sol) {
            std::cout << "failure" << std::endl;
        } else {
            std::cout << "success" << std::endl;
        }
    }
    // =====================================================================================
    std::cout << "Testing scan_gpu1() ________________" << std::endl;
    {
        std::cout << "Test 1 ";
        const std::vector<int> x = {3,2,5,6,8,7,4,1};
        const std::vector<int> y_sol = {0,3,5,10,16,24,31,35};
        const std::vector<int> y_test = scan_gpu1<8>(x);
        if(y_test != y_sol) {
            std::cout << "failure" << std::endl;
        } else {
            std::cout << "success" << std::endl;
        }
    }
    {
        std::cout << "Test 2 ";
        const std::vector<int> x = {16,15,4,14,10,1,13,12,2,11,9,7,8,5,3,6};
        const std::vector<int> y_sol = {0,16,31,35,49,59,60,73,85,87,98,107,114,122,127,130};
        const std::vector<int> y_test = scan_gpu1<16>(x);
        if(y_test != y_sol) {
            std::cout << "failure" << std::endl;
        } else {
            std::cout << "success" << std::endl;
        }
    }
    {
        std::cout << "Test 3 ";
        const std::vector<int> x = {632,680,365,320,488,724,484,955,252,271,76,672,679,321,368,295,857,690,507,531,2,28,532,32,514,547,963,852,381,313,332,944,19,919,810,866,519,1016,296,404,279,215,614,863,123,318,562,15,16,143,913,86,197,387,248,357,193,996,142,559,440,433,972,485,262,46,838,760,795,743,635,56,1001,968,410,639,109,407,463,952,744,787,78,352,581,414,586,630,833,993,957,419,584,763,974,284,98,331,317,468,1014,990,104,379,231,748,396,3,436,667,652,426,176,776,304,656,842,577,201,906,829,448,890,751,583,203,747,730,256,716,300,821,861,722,814,254,605,774,421,66,650,801,699,647,615,311,265,437,309,61,316,425,641,846,626,512,960,770,878,495,849,527,453,250,9,560,860,102,268,501,435,794,253,454,726,643,444,55,35,240,31,791,796,649,688,479,870,826,905,523,958,912,725,956,480,521,684,892,831,903,191,280,534,836,146,859,478,773,628,278,813,671,708,145,879,766,373,400,666,216,694,703,754,343,403,441,385,155,205,683,6,589,732,997,759,417,689,151,232,978,627,588,492,554,604,273,546,458,241,777,929,114,761,38,719,696,943,159,41,359,486,353,568,921,237,633,502,644,799,558,362,496,63,80,194,409,182,360,610,10,845,835,494,733,263,7,267,971,39,995,762,299,272,141,651,195,753,659,809,466,742,616,51,163,530,312,413,876,579,837,173,940,367,818,758,983,805,40,593,825,157,223,677,211,307,784,1013,737,168,966,789,383,11,106,69,91,475,487,259,597,973,305,936,549,989,975,209,21,621,953,423,827,286,662,72,50,180,856,363,117,294,77,49,398,264,850,811,950,333,65,839,53,364,493,125,686,459,472,1004,524,303,119,13,158,338,128,47,236,569,242,619,490,959,1011,893,908,326,947,823,351,636,18,1020,187,452,339,207,483,920,346,314,243,482,171,188,949,134,302,768,642,319,79,819,347,397,199,390,386,969,467,965,620,824,550,769,22,681,755,600,251,244,602,36,82,160,718,334,881,497,707,110,631,380,704,565,93,504,522,999,606,391,609,71,473,212,172,166,239,189,1012,665,655,345,572,930,506,682,874,156,525,603,793,126,457,67,645,234,225,269,162,625,539,712,179,734,416,101,570,657,230,335,942,739,802,914,970,221,392,1003,428,310,815,932,528,246,670,786,105,1023,52,355,461,23,375,219,536,904,954,206,709,498,517,376,186,405,901,402,935,692,442,133,174,429,598,1021,797,1008,97,623,500,75,90,790,601,1,393,477,840,520,108,745,668,224,328,103,741,476,214,136,247,1024,1010,653,948,366,855,27,678,418,673,591,4,135,356,923,808,880,822,832,226,864,370,804,340,420,887,1002,455,64,1009,361,238,113,931,533,578,127,260,816,746,138,8,503,875,738,896,408,729,354,154,567,543,344,934,613,721,946,57,401,573,474,469,687,705,505,675,685,982,301,800,701,571,164,637,337,60,865,984,491,828,450,900,438,564,980,937,220,714,587,283,349,723,5,1017,464,922,693,926,518,654,574,648,834,941,535,951,277,285,471,427,282,121,594,315,14,924,456,585,1007,161,88,713,752,782,196,192,629,537,927,165,107,17,576,976,749,541,330,481,499,916,406,781,94,434,84,888,820,711,658,415,663,552,767,765,961,788,323,557,358,70,515,510,341,222,979,147,290,30,115,509,853,538,177,169,348,377,988,1022,898,812,190,137,447,991,255,928,291,992,891,233,646,34,89,45,342,62,1005,200,430,785,395,806,731,298,74,394,548,580,542,899,877,884,664,218,540,867,872,624,669,868,422,695,139,213,617,886,925,1006,728,1018,702,727,967,717,235,772,68,25,382,217,736,130,350,112,167,964,208,439,553,275,894,592,858,843,526,288,470,327,122,764,508,229,674,596,446,118,465,26,111,150,153,529,697,844,622,462,599,140,750,261,895,830,324,945,249,451,411,841,608,20,329,297,270,227,95,516,885,638,281,287,590,981,445,185,372,918,175,276,807,986,183,33,81,308,771,12,873,44,87,779,37,706,59,42,783,412,563,210,149,740,595,124,735,575,99,378,131,700,24,915,556,371,907,513,384,202,369,555,611,388,862,817,293,691,882,257,607,198,96,1015,994,170,582,306,561,854,374,910,720,792,228,661,144,847,181,869,132,1000,48,775,325,204,85,939,634,909,566,612,399,897,92,83,431,780,120,871,292,58,938,322,258,889,998,443,43,618,389,336,274,757,489,551,902,54,933,987,129,803,29,184,883,460,511,676,152,245,715,544,545,640,698,266,962,289,424,911,148,660,798,1019,178,985,100,851,449,432,848,710,756,977,116,917,778,73};
        const std::vector<int> y_sol = {0,632,1312,1677,1997,2485,3209,3693,4648,4900,5171,5247,5919,6598,6919,7287,7582,8439,9129,9636,10167,10169,10197,10729,10761,11275,11822,12785,13637,14018,14331,14663,15607,15626,16545,17355,18221,18740,19756,20052,20456,20735,20950,21564,22427,22550,22868,23430,23445,23461,23604,24517,24603,24800,25187,25435,25792,25985,26981,27123,27682,28122,28555,29527,30012,30274,30320,31158,31918,32713,33456,34091,34147,35148,36116,36526,37165,37274,37681,38144,39096,39840,40627,40705,41057,41638,42052,42638,43268,44101,45094,46051,46470,47054,47817,48791,49075,49173,49504,49821,50289,51303,52293,52397,52776,53007,53755,54151,54154,54590,55257,55909,56335,56511,57287,57591,58247,59089,59666,59867,60773,61602,62050,62940,63691,64274,64477,65224,65954,66210,66926,67226,68047,68908,69630,70444,70698,71303,72077,72498,72564,73214,74015,74714,75361,75976,76287,76552,76989,77298,77359,77675,78100,78741,79587,80213,80725,81685,82455,83333,83828,84677,85204,85657,85907,85916,86476,87336,87438,87706,88207,88642,89436,89689,90143,90869,91512,91956,92011,92046,92286,92317,93108,93904,94553,95241,95720,96590,97416,98321,98844,99802,100714,101439,102395,102875,103396,104080,104972,105803,106706,106897,107177,107711,108547,108693,109552,110030,110803,111431,111709,112522,113193,113901,114046,114925,115691,116064,116464,117130,117346,118040,118743,119497,119840,120243,120684,121069,121224,121429,122112,122118,122707,123439,124436,125195,125612,126301,126452,126684,127662,128289,128877,129369,129923,130527,130800,131346,131804,132045,132822,133751,133865,134626,134664,135383,136079,137022,137181,137222,137581,138067,138420,138988,139909,140146,140779,141281,141925,142724,143282,143644,144140,144203,144283,144477,144886,145068,145428,146038,146048,146893,147728,148222,148955,149218,149225,149492,150463,150502,151497,152259,152558,152830,152971,153622,153817,154570,155229,156038,156504,157246,157862,157913,158076,158606,158918,159331,160207,160786,161623,161796,162736,163103,163921,164679,165662,166467,166507,167100,167925,168082,168305,168982,169193,169500,170284,171297,172034,172202,173168,173957,174340,174351,174457,174526,174617,175092,175579,175838,176435,177408,177713,178649,179198,180187,181162,181371,181392,182013,182966,183389,184216,184502,185164,185236,185286,185466,186322,186685,186802,187096,187173,187222,187620,187884,188734,189545,190495,190828,190893,191732,191785,192149,192642,192767,193453,193912,194384,195388,195912,196215,196334,196347,196505,196843,196971,197018,197254,197823,198065,198684,199174,200133,201144,202037,202945,203271,204218,205041,205392,206028,206046,207066,207253,207705,208044,208251,208734,209654,210000,210314,210557,211039,211210,211398,212347,212481,212783,213551,214193,214512,214591,215410,215757,216154,216353,216743,217129,218098,218565,219530,220150,220974,221524,222293,222315,222996,223751,224351,224602,224846,225448,225484,225566,225726,226444,226778,227659,228156,228863,228973,229604,229984,230688,231253,231346,231850,232372,233371,233977,234368,234977,235048,235521,235733,235905,236071,236310,236499,237511,238176,238831,239176,239748,240678,241184,241866,242740,242896,243421,244024,244817,244943,245400,245467,246112,246346,246571,246840,247002,247627,248166,248878,249057,249791,250207,250308,250878,251535,251765,252100,253042,253781,254583,255497,256467,256688,257080,258083,258511,258821,259636,260568,261096,261342,262012,262798,262903,263926,263978,264333,264794,264817,265192,265411,265947,266851,267805,268011,268720,269218,269735,270111,270297,270702,271603,272005,272940,273632,274074,274207,274381,274810,275408,276429,277226,278234,278331,278954,279454,279529,279619,280409,281010,281011,281404,281881,282721,283241,283349,284094,284762,284986,285314,285417,286158,286634,286848,286984,287231,288255,289265,289918,290866,291232,292087,292114,292792,293210,293883,294474,294478,294613,294969,295892,296700,297580,298402,299234,299460,300324,300694,301498,301838,302258,303145,304147,304602,304666,305675,306036,306274,306387,307318,307851,308429,308556,308816,309632,310378,310516,310524,311027,311902,312640,313536,313944,314673,315027,315181,315748,316291,316635,317569,318182,318903,319849,319906,320307,320880,321354,321823,322510,323215,323720,324395,325080,326062,326363,327163,327864,328435,328599,329236,329573,329633,330498,331482,331973,332801,333251,334151,334589,335153,336133,337070,337290,338004,338591,338874,339223,339946,339951,340968,341432,342354,343047,343973,344491,345145,345719,346367,347201,348142,348677,349628,349905,350190,350661,351088,351370,351491,352085,352400,352414,353338,353794,354379,355386,355547,355635,356348,357100,357882,358078,358270,358899,359436,360363,360528,360635,360652,361228,362204,362953,363494,363824,364305,364804,365720,366126,366907,367001,367435,367519,368407,369227,369938,370596,371011,371674,372226,372993,373758,374719,375507,375830,376387,376745,376815,377330,377840,378181,378403,379382,379529,379819,379849,379964,380473,381326,381864,382041,382210,382558,382935,383923,384945,385843,386655,386845,386982,387429,388420,388675,389603,389894,390886,391777,392010,392656,392690,392779,392824,393166,393228,394233,394433,394863,395648,396043,396849,397580,397878,397952,398346,398894,399474,400016,400915,401792,402676,403340,403558,404098,404965,405837,406461,407130,407998,408420,409115,409254,409467,410084,410970,411895,412901,413629,414647,415349,416076,417043,417760,417995,418767,418835,418860,419242,419459,420195,420325,420675,420787,420954,421918,422126,422565,423118,423393,424287,424879,425737,426580,427106,427394,427864,428191,428313,429077,429585,429814,430488,431084,431530,431648,432113,432139,432250,432400,432553,433082,433779,434623,435245,435707,436306,436446,437196,437457,438352,439182,439506,440451,440700,441151,441562,442403,443011,443031,443360,443657,443927,444154,444249,444765,445650,446288,446569,446856,447446,448427,448872,449057,449429,450347,450522,450798,451605,452591,452774,452807,452888,453196,453967,453979,454852,454896,454983,455762,455799,456505,456564,456606,457389,457801,458364,458574,458723,459463,460058,460182,460917,461492,461591,461969,462100,462800,462824,463739,464295,464666,465573,466086,466470,466672,467041,467596,468207,468595,469457,470274,470567,471258,472140,472397,473004,473202,473298,474313,475307,475477,476059,476365,476926,477780,478154,479064,479784,480576,480804,481465,481609,482456,482637,483506,483638,484638,484686,485461,485786,485990,486075,487014,487648,488557,489123,489735,490134,491031,491123,491206,491637,492417,492537,493408,493700,493758,494696,495018,495276,496165,497163,497606,497649,498267,498656,498992,499266,500023,500512,501063,501965,502019,502952,503939,504068,504871,504900,505084,505967,506427,506938,507614,507766,508011,508726,509270,509815,510455,511153,511419,512381,512670,513094,514005,514153,514813,515611,516630,516808,517793,517893,518744,519193,519625,520473,521183,521939,522916,523032,523949,524727};
        const std::vector<int> y_test = scan_gpu1<1024>(x);
        if(y_test != y_sol) {
            std::cout << "failure" << std::endl;
        } else {
            std::cout << "success" << std::endl;
        }
    }
    // =====================================================================================
    std::cout << "Testing scan_gpu2() ________________" << std::endl;
    {
        std::cout << "Test 1 ";
        const std::vector<int> x = {3,2,5,6,8,7,4,1};
        const std::vector<int> y_sol = {0,3,5,10,16,24,31,35};
        const std::vector<int> y_test = scan_gpu2<8>(x);
        if(y_test != y_sol) {
            std::cout << "failure" << std::endl;
        } else {
            std::cout << "success" << std::endl;
        }
    }
    {
        std::cout << "Test 2 ";
        const std::vector<int> x = {16,15,4,14,10,1,13,12,2,11,9,7,8,5,3,6};
        const std::vector<int> y_sol = {0,16,31,35,49,59,60,73,85,87,98,107,114,122,127,130};
        const std::vector<int> y_test = scan_gpu2<16>(x);
        if(y_test != y_sol) {
            std::cout << "failure" << std::endl;
        } else {
            std::cout << "success" << std::endl;
        }
    }
    {
        std::cout << "Test 3 ";
        const std::vector<int> x = {632,680,365,320,488,724,484,955,252,271,76,672,679,321,368,295,857,690,507,531,2,28,532,32,514,547,963,852,381,313,332,944,19,919,810,866,519,1016,296,404,279,215,614,863,123,318,562,15,16,143,913,86,197,387,248,357,193,996,142,559,440,433,972,485,262,46,838,760,795,743,635,56,1001,968,410,639,109,407,463,952,744,787,78,352,581,414,586,630,833,993,957,419,584,763,974,284,98,331,317,468,1014,990,104,379,231,748,396,3,436,667,652,426,176,776,304,656,842,577,201,906,829,448,890,751,583,203,747,730,256,716,300,821,861,722,814,254,605,774,421,66,650,801,699,647,615,311,265,437,309,61,316,425,641,846,626,512,960,770,878,495,849,527,453,250,9,560,860,102,268,501,435,794,253,454,726,643,444,55,35,240,31,791,796,649,688,479,870,826,905,523,958,912,725,956,480,521,684,892,831,903,191,280,534,836,146,859,478,773,628,278,813,671,708,145,879,766,373,400,666,216,694,703,754,343,403,441,385,155,205,683,6,589,732,997,759,417,689,151,232,978,627,588,492,554,604,273,546,458,241,777,929,114,761,38,719,696,943,159,41,359,486,353,568,921,237,633,502,644,799,558,362,496,63,80,194,409,182,360,610,10,845,835,494,733,263,7,267,971,39,995,762,299,272,141,651,195,753,659,809,466,742,616,51,163,530,312,413,876,579,837,173,940,367,818,758,983,805,40,593,825,157,223,677,211,307,784,1013,737,168,966,789,383,11,106,69,91,475,487,259,597,973,305,936,549,989,975,209,21,621,953,423,827,286,662,72,50,180,856,363,117,294,77,49,398,264,850,811,950,333,65,839,53,364,493,125,686,459,472,1004,524,303,119,13,158,338,128,47,236,569,242,619,490,959,1011,893,908,326,947,823,351,636,18,1020,187,452,339,207,483,920,346,314,243,482,171,188,949,134,302,768,642,319,79,819,347,397,199,390,386,969,467,965,620,824,550,769,22,681,755,600,251,244,602,36,82,160,718,334,881,497,707,110,631,380,704,565,93,504,522,999,606,391,609,71,473,212,172,166,239,189,1012,665,655,345,572,930,506,682,874,156,525,603,793,126,457,67,645,234,225,269,162,625,539,712,179,734,416,101,570,657,230,335,942,739,802,914,970,221,392,1003,428,310,815,932,528,246,670,786,105,1023,52,355,461,23,375,219,536,904,954,206,709,498,517,376,186,405,901,402,935,692,442,133,174,429,598,1021,797,1008,97,623,500,75,90,790,601,1,393,477,840,520,108,745,668,224,328,103,741,476,214,136,247,1024,1010,653,948,366,855,27,678,418,673,591,4,135,356,923,808,880,822,832,226,864,370,804,340,420,887,1002,455,64,1009,361,238,113,931,533,578,127,260,816,746,138,8,503,875,738,896,408,729,354,154,567,543,344,934,613,721,946,57,401,573,474,469,687,705,505,675,685,982,301,800,701,571,164,637,337,60,865,984,491,828,450,900,438,564,980,937,220,714,587,283,349,723,5,1017,464,922,693,926,518,654,574,648,834,941,535,951,277,285,471,427,282,121,594,315,14,924,456,585,1007,161,88,713,752,782,196,192,629,537,927,165,107,17,576,976,749,541,330,481,499,916,406,781,94,434,84,888,820,711,658,415,663,552,767,765,961,788,323,557,358,70,515,510,341,222,979,147,290,30,115,509,853,538,177,169,348,377,988,1022,898,812,190,137,447,991,255,928,291,992,891,233,646,34,89,45,342,62,1005,200,430,785,395,806,731,298,74,394,548,580,542,899,877,884,664,218,540,867,872,624,669,868,422,695,139,213,617,886,925,1006,728,1018,702,727,967,717,235,772,68,25,382,217,736,130,350,112,167,964,208,439,553,275,894,592,858,843,526,288,470,327,122,764,508,229,674,596,446,118,465,26,111,150,153,529,697,844,622,462,599,140,750,261,895,830,324,945,249,451,411,841,608,20,329,297,270,227,95,516,885,638,281,287,590,981,445,185,372,918,175,276,807,986,183,33,81,308,771,12,873,44,87,779,37,706,59,42,783,412,563,210,149,740,595,124,735,575,99,378,131,700,24,915,556,371,907,513,384,202,369,555,611,388,862,817,293,691,882,257,607,198,96,1015,994,170,582,306,561,854,374,910,720,792,228,661,144,847,181,869,132,1000,48,775,325,204,85,939,634,909,566,612,399,897,92,83,431,780,120,871,292,58,938,322,258,889,998,443,43,618,389,336,274,757,489,551,902,54,933,987,129,803,29,184,883,460,511,676,152,245,715,544,545,640,698,266,962,289,424,911,148,660,798,1019,178,985,100,851,449,432,848,710,756,977,116,917,778,73};
        const std::vector<int> y_sol = {0,632,1312,1677,1997,2485,3209,3693,4648,4900,5171,5247,5919,6598,6919,7287,7582,8439,9129,9636,10167,10169,10197,10729,10761,11275,11822,12785,13637,14018,14331,14663,15607,15626,16545,17355,18221,18740,19756,20052,20456,20735,20950,21564,22427,22550,22868,23430,23445,23461,23604,24517,24603,24800,25187,25435,25792,25985,26981,27123,27682,28122,28555,29527,30012,30274,30320,31158,31918,32713,33456,34091,34147,35148,36116,36526,37165,37274,37681,38144,39096,39840,40627,40705,41057,41638,42052,42638,43268,44101,45094,46051,46470,47054,47817,48791,49075,49173,49504,49821,50289,51303,52293,52397,52776,53007,53755,54151,54154,54590,55257,55909,56335,56511,57287,57591,58247,59089,59666,59867,60773,61602,62050,62940,63691,64274,64477,65224,65954,66210,66926,67226,68047,68908,69630,70444,70698,71303,72077,72498,72564,73214,74015,74714,75361,75976,76287,76552,76989,77298,77359,77675,78100,78741,79587,80213,80725,81685,82455,83333,83828,84677,85204,85657,85907,85916,86476,87336,87438,87706,88207,88642,89436,89689,90143,90869,91512,91956,92011,92046,92286,92317,93108,93904,94553,95241,95720,96590,97416,98321,98844,99802,100714,101439,102395,102875,103396,104080,104972,105803,106706,106897,107177,107711,108547,108693,109552,110030,110803,111431,111709,112522,113193,113901,114046,114925,115691,116064,116464,117130,117346,118040,118743,119497,119840,120243,120684,121069,121224,121429,122112,122118,122707,123439,124436,125195,125612,126301,126452,126684,127662,128289,128877,129369,129923,130527,130800,131346,131804,132045,132822,133751,133865,134626,134664,135383,136079,137022,137181,137222,137581,138067,138420,138988,139909,140146,140779,141281,141925,142724,143282,143644,144140,144203,144283,144477,144886,145068,145428,146038,146048,146893,147728,148222,148955,149218,149225,149492,150463,150502,151497,152259,152558,152830,152971,153622,153817,154570,155229,156038,156504,157246,157862,157913,158076,158606,158918,159331,160207,160786,161623,161796,162736,163103,163921,164679,165662,166467,166507,167100,167925,168082,168305,168982,169193,169500,170284,171297,172034,172202,173168,173957,174340,174351,174457,174526,174617,175092,175579,175838,176435,177408,177713,178649,179198,180187,181162,181371,181392,182013,182966,183389,184216,184502,185164,185236,185286,185466,186322,186685,186802,187096,187173,187222,187620,187884,188734,189545,190495,190828,190893,191732,191785,192149,192642,192767,193453,193912,194384,195388,195912,196215,196334,196347,196505,196843,196971,197018,197254,197823,198065,198684,199174,200133,201144,202037,202945,203271,204218,205041,205392,206028,206046,207066,207253,207705,208044,208251,208734,209654,210000,210314,210557,211039,211210,211398,212347,212481,212783,213551,214193,214512,214591,215410,215757,216154,216353,216743,217129,218098,218565,219530,220150,220974,221524,222293,222315,222996,223751,224351,224602,224846,225448,225484,225566,225726,226444,226778,227659,228156,228863,228973,229604,229984,230688,231253,231346,231850,232372,233371,233977,234368,234977,235048,235521,235733,235905,236071,236310,236499,237511,238176,238831,239176,239748,240678,241184,241866,242740,242896,243421,244024,244817,244943,245400,245467,246112,246346,246571,246840,247002,247627,248166,248878,249057,249791,250207,250308,250878,251535,251765,252100,253042,253781,254583,255497,256467,256688,257080,258083,258511,258821,259636,260568,261096,261342,262012,262798,262903,263926,263978,264333,264794,264817,265192,265411,265947,266851,267805,268011,268720,269218,269735,270111,270297,270702,271603,272005,272940,273632,274074,274207,274381,274810,275408,276429,277226,278234,278331,278954,279454,279529,279619,280409,281010,281011,281404,281881,282721,283241,283349,284094,284762,284986,285314,285417,286158,286634,286848,286984,287231,288255,289265,289918,290866,291232,292087,292114,292792,293210,293883,294474,294478,294613,294969,295892,296700,297580,298402,299234,299460,300324,300694,301498,301838,302258,303145,304147,304602,304666,305675,306036,306274,306387,307318,307851,308429,308556,308816,309632,310378,310516,310524,311027,311902,312640,313536,313944,314673,315027,315181,315748,316291,316635,317569,318182,318903,319849,319906,320307,320880,321354,321823,322510,323215,323720,324395,325080,326062,326363,327163,327864,328435,328599,329236,329573,329633,330498,331482,331973,332801,333251,334151,334589,335153,336133,337070,337290,338004,338591,338874,339223,339946,339951,340968,341432,342354,343047,343973,344491,345145,345719,346367,347201,348142,348677,349628,349905,350190,350661,351088,351370,351491,352085,352400,352414,353338,353794,354379,355386,355547,355635,356348,357100,357882,358078,358270,358899,359436,360363,360528,360635,360652,361228,362204,362953,363494,363824,364305,364804,365720,366126,366907,367001,367435,367519,368407,369227,369938,370596,371011,371674,372226,372993,373758,374719,375507,375830,376387,376745,376815,377330,377840,378181,378403,379382,379529,379819,379849,379964,380473,381326,381864,382041,382210,382558,382935,383923,384945,385843,386655,386845,386982,387429,388420,388675,389603,389894,390886,391777,392010,392656,392690,392779,392824,393166,393228,394233,394433,394863,395648,396043,396849,397580,397878,397952,398346,398894,399474,400016,400915,401792,402676,403340,403558,404098,404965,405837,406461,407130,407998,408420,409115,409254,409467,410084,410970,411895,412901,413629,414647,415349,416076,417043,417760,417995,418767,418835,418860,419242,419459,420195,420325,420675,420787,420954,421918,422126,422565,423118,423393,424287,424879,425737,426580,427106,427394,427864,428191,428313,429077,429585,429814,430488,431084,431530,431648,432113,432139,432250,432400,432553,433082,433779,434623,435245,435707,436306,436446,437196,437457,438352,439182,439506,440451,440700,441151,441562,442403,443011,443031,443360,443657,443927,444154,444249,444765,445650,446288,446569,446856,447446,448427,448872,449057,449429,450347,450522,450798,451605,452591,452774,452807,452888,453196,453967,453979,454852,454896,454983,455762,455799,456505,456564,456606,457389,457801,458364,458574,458723,459463,460058,460182,460917,461492,461591,461969,462100,462800,462824,463739,464295,464666,465573,466086,466470,466672,467041,467596,468207,468595,469457,470274,470567,471258,472140,472397,473004,473202,473298,474313,475307,475477,476059,476365,476926,477780,478154,479064,479784,480576,480804,481465,481609,482456,482637,483506,483638,484638,484686,485461,485786,485990,486075,487014,487648,488557,489123,489735,490134,491031,491123,491206,491637,492417,492537,493408,493700,493758,494696,495018,495276,496165,497163,497606,497649,498267,498656,498992,499266,500023,500512,501063,501965,502019,502952,503939,504068,504871,504900,505084,505967,506427,506938,507614,507766,508011,508726,509270,509815,510455,511153,511419,512381,512670,513094,514005,514153,514813,515611,516630,516808,517793,517893,518744,519193,519625,520473,521183,521939,522916,523032,523949,524727};
        const std::vector<int> y_test = scan_gpu2<1024>(x);
        if(y_test != y_sol) {
            std::cout << "failure" << std::endl;
        } else {
            std::cout << "success" << std::endl;
        }
    }

    return 0;
}