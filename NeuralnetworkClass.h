//
//  NeuralnetworkClass.h
//  Neuralnetwork
//
//  Created by 遠藤 豪 on 2014/02/17.
//  Copyright (c) 2014年 endo.neural. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface NeuralnetworkClass : NSObject
@property (nonatomic) NSMutableArray *arrInput, *arrHidden, *arrOutput;
@property (nonatomic) NSMutableArray *arrError;
@property (nonatomic) NSMutableArray *arrHiddenDelta;
@property (nonatomic) NSMutableArray *arrOutputDelta;
@property (nonatomic) NSMutableArray *arrStatusOfHidden, *arrStatusOfOutput;
@property (nonatomic) NSMutableArray *arrWeightIH, *arrWeightHO;
@property (nonatomic) NSMutableArray *arrDeltaWeightIH, *arrDeltaWeightHO;
@property (nonatomic) NSMutableArray *arrSupervisor;
-(id)initWithInput:(int)_numOfInput
        withHidden:(int)_numOfHidden
        withOutput:(int)_numOfOutput;
@end
