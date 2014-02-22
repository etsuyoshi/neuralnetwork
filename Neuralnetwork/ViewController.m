//
//  ViewController.m
//  Neuralnetwork
//
//  Created by 遠藤 豪 on 2014/02/17.
//  Copyright (c) 2014年 endo.neural. All rights reserved.
//

#import "ViewController.h"
#import "NeuralnetworkClass.h"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad
{
    [super viewDidLoad];
	// Do any additional setup after loading the view, typically from a nib.
}

- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

-(void)viewDidAppear:(BOOL)animated{
    NeuralnetworkClass *nn =
    [[NeuralnetworkClass alloc]
     initWithInput:4 withHidden:4 withOutput:3];
    
    
    NSMutableArray *arrMyInput =
    [NSMutableArray arrayWithObjects:
     
     [NSMutableArray arrayWithObjects:
      [NSNumber numberWithDouble:1.5f],
      [NSNumber numberWithDouble:0.0f],
      [NSNumber numberWithDouble:1.0f],
      [NSNumber numberWithDouble:0.0f],
      nil],
     [NSMutableArray arrayWithObjects:
      [NSNumber numberWithDouble:1.5f],
      [NSNumber numberWithDouble:1.0f],
      [NSNumber numberWithDouble:0.0f],
      [NSNumber numberWithDouble:1.0f],
      nil],
     nil];
    
    NSMutableArray *arrMySupervisor =
    [NSMutableArray arrayWithObjects:
     [NSMutableArray arrayWithObjects:
      [NSNumber numberWithDouble:0.5f],
      [NSNumber numberWithDouble:1.0f],
      [NSNumber numberWithDouble:0.3f],
      nil],
     [NSMutableArray arrayWithObjects:
      [NSNumber numberWithDouble:1.0f],
      [NSNumber numberWithDouble:0.8f],
      [NSNumber numberWithDouble:0.3f],
      nil],
     nil];
    
//    nn.arrSupervisor =
//    [NSMutableArray arrayWithObjects:
//     [NSNumber numberWithDouble:0.5f],
//     [NSNumber numberWithDouble:1.0f],
//     [NSNumber numberWithDouble:0.3f],
//     nil];
    
    
    for(int t = 0;t < 10000;t++){
        BOOL isCompleteFF =
        [nn forwardWithInput:arrMyInput[t%2]];
//         [NSMutableArray arrayWithObjects:
//          [NSNumber numberWithDouble:1.5f],
//          [NSNumber numberWithDouble:1.0f],
//          [NSNumber numberWithDouble:0.5f],
//          [NSNumber numberWithDouble:1.0f],
//          nil]];
        nn.arrSupervisor = arrMySupervisor[t%2];
        
        BOOL isCompleteBP = [nn backPropagation];
        
//        NSLog(@"ff=%@, bp=%@",
//              isCompleteFF?@"成功":@"失敗",
//              isCompleteBP?@"成功":@"失敗");
        if(!(isCompleteFF && isCompleteBP)){
            NSLog(@"error occurring!! at %@",
                  isCompleteFF?@"BP":@"FF");
            break;
        }
    }
    
    NSLog(@"learning complete");
    
    NSMutableArray *_arrOutput1 = nn.arrOutput;
    for(int k = 0;k < [_arrOutput1 count];k++){
        NSLog(@"output%d is %f", k, [_arrOutput1[k] doubleValue]);
    }
    
    
    //test
    [nn forwardWithInput:arrMyInput[0]];
    for(int k = 0;k < [nn.arrOutput count];k++){
        NSLog(@"pattern1:output%d is %f against:%f",
              k, [nn.arrOutput[k] doubleValue],
              [arrMySupervisor[0][k] doubleValue]);
    }
    
    
    [nn forwardWithInput:arrMyInput[1]];
    for(int k = 0;k < [nn.arrOutput count];k++){
        NSLog(@"pattern1:output%d is %f against:%f",
              k, [nn.arrOutput[k] doubleValue],
              [arrMySupervisor[1][k] doubleValue]);
    }
    
    
    
}



@end
