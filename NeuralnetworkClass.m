//
//  NeuralnetworkClass.m
//  Neuralnetwork
//
//  Created by 遠藤 豪 on 2014/02/17.
//  Copyright (c) 2014年 endo.neural. All rights reserved.
//

#import "NeuralnetworkClass.h"

#define ETA 0.5


@implementation NeuralnetworkClass

//NSMutableArray *arrInput, *arrHidden, *arrOutput;
//NSMutableArray *arrWeightIH, *arrWeightHO;
@synthesize arrInput;
@synthesize arrHidden;
@synthesize arrOutput;
@synthesize arrStatusOfHidden;//内部状態：前の層からの結合加重を通って渡される値そのもの
@synthesize arrStatusOfOutput;
@synthesize arrWeightIH;
@synthesize arrWeightHO;
@synthesize arrDeltaWeightIH;
@synthesize arrDeltaWeightHO;
@synthesize arrSupervisor;

@synthesize arrError;//教師信号に対する誤差
@synthesize arrHiddenDelta;//中間層ニューロンデルタ
@synthesize arrOutputDelta;//出力層ニューロンデルタ


-(id)initWithInput:(int)_numOfInput
        withHidden:(int)_numOfHidden
        withOutput:(int)_numOfOutput{
    self = [super init];
    
    if(self){
        //入力層初期化
        self.arrInput = [NSMutableArray array];
        for(int i = 0;i < _numOfInput;i++){
            [self.arrInput addObject:[NSNumber numberWithDouble:0]];
        }
        
        //中間層初期化
        self.arrHidden = [NSMutableArray array];
        self.arrHiddenDelta = [NSMutableArray array];
        for(int h = 0;h < _numOfHidden;h++){
            [self.arrHidden addObject:[NSNumber numberWithDouble:0]];
            [self.arrStatusOfHidden addObject:[NSNumber numberWithDouble:0]];
            
            [self.arrHiddenDelta addObject:[NSNumber numberWithDouble:0]];
        }
        
        //出力層初期化
        self.arrOutput = [NSMutableArray array];
        for(int o = 0;o < _numOfOutput;o++){
            [self.arrOutput addObject:[NSNumber numberWithDouble:0]];
            [self.arrStatusOfOutput addObject:[NSNumber numberWithDouble:0]];
        }
        
        //結合加重：入力層ー中間層
        NSMutableArray *arrWeightTmp;
        float fRand = 0;
        for(int i = 0;i < _numOfOutput;i++){
            arrWeightTmp = [NSMutableArray array];
            for(int h = 0;h < _numOfHidden;h++){
                fRand = (float)(arc4random() % 100)/100.0f;
                NSLog(@"i,h=%d,%d of weight = %f",i,h, fRand);
                [arrWeightTmp addObject:[NSNumber numberWithDouble:fRand]];
            }
            [arrWeightIH addObject:arrWeightTmp];
        }
        
        //結合加重：中間層ー出力層
        for(int h =0;h < _numOfHidden;h++){
            arrWeightTmp = [NSMutableArray array];
            for(int o = 0;o < _numOfOutput;o++){
                fRand = (float)(arc4random() % 100)/100.0f;
                NSLog(@"h,o=%d,%d of weight = %f",h,o, fRand);
                [arrOutput addObject:[NSNumber numberWithDouble:fRand]];
            }
            [arrWeightHO addObject:arrWeightTmp];
        }
        
        
    }
    
    
    return self;
}

-(BOOL)forwardWithInput:(NSMutableArray *)_arrInputValue{
    if([self.arrInput count] != [_arrInputValue count]){
        NSLog(@"inputNeuron : number of neuron error");
        return false;
    }
    
    //入力値の指定
    for(int i = 0;i < [_arrInputValue count];i++){
        self.arrInput[i] =
        [NSNumber numberWithDouble:
         [_arrInputValue[i] doubleValue]];
    }
    
    //中間層の計算
    double _statusOfNeuron = 0;
    for(int h = 0;h < [self.arrHidden count];h++){
        //中間層の内部状態の計算
        _statusOfNeuron = 0;
        for(int i = 0;i < [self.arrInput count];i++){
            _statusOfNeuron +=
            [self.arrInput[i] doubleValue] *
            [self.arrWeightIH[i][h] doubleValue];
        }
        
        //中間層出力値の計算
        arrHidden[h] =
        [NSNumber numberWithDouble:
         [self getSigmoidHidden:_statusOfNeuron]];
        
//        NSLog(@"output of hidden %d is %f",
//              h, [self getSigmoidHidden:_statusOfNeuron]);
    }
    
    //出力層の計算
    for(int o = 0;o < [self.arrOutput count];o++){
        //出力層の内部状態の計算
        _statusOfNeuron = 0;
        for(int h = 0;h < [self.arrHidden count];h++){
            _statusOfNeuron +=
            [self.arrHidden[h] doubleValue] *
            [self.arrWeightHO[h][o] doubleValue];
        }
        
        //出力層出力値の計算
        arrOutput[o] =
        [NSNumber numberWithDouble:
         [self getSigmoidOutput:_statusOfNeuron]];
        
        
    }
    
    return true;
}


-(BOOL)backPropagation{
    if([self.arrSupervisor count] != [self.arrOutput count]){
        return false;
    }
    
    double _error = 0;
    double _supervisor = 0;
    double _output = 0;
    self.arrError = [NSMutableArray array];
    self.arrOutputDelta = [NSMutableArray array];
    for(int o = 0;o < [self.arrOutput count];o++){
        _supervisor = [self.arrSupervisor[o] doubleValue];
        _output = [self.arrOutput[o] doubleValue];
        _error = _supervisor - _output;
        //http://www.fer.unizg.hr/_download/repository/BP_chapter3_-_bp.pdf
//        _error =
//        [self.arrOutput[o] doubleValue] * (1- [self.arrOutput[o] doubleValue]) *
//        ([self.arrOutput[o] doubleValue]  -
//         [self.arrSupervisor[o] doubleValue]);
        
        [arrError addObject:[NSNumber numberWithDouble:_error]];
        
        [arrOutputDelta addObject:
         [NSNumber numberWithDouble:
          _output * (1 - _output) * (_supervisor - _output)]];
    }
    
    
    //中間層j-出力層k間の結合加重の修正量
    //deltaWkj = eta * delta_k * Hj(j番目中間層出力値)
    //ただし、delta_k=(supervisor_k - output_k) * output_k * (1 - output_k)
    self.arrDeltaWeightHO = [NSMutableArray array];
    for(int j = 0;j < [arrHidden count];j++){
        for(int k = 0;k < [arrOutput count];k++){
            
        }
    }
    
    
    
    //入力層i-中間層j間の結合加重の修正量
    //deltaWji = eta * Hj * (1 - Hj) * Σ(wkj * delta_k)
    //delta_k=(supervisor_k - output_k) * output_k * (1 - output_k)
    
    
    return true;
}



-(double)getSigmoidHidden:(double)_status{
    double parameter = 1.0f;
    double denominator = 1.0f + exp(-parameter * _status + 0.5f);
    return 1.0f/denominator;
}

-(double)getSigmoidOutput:(double)_status{
    double parameter = 1.0f;
    double denominator = 1.0f + exp(-parameter * _status + 0.5f);
    return 1.0f/denominator;
}


@end
