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
     initWithInput:3 withHidden:4 withOutput:3];
    
}

@end
