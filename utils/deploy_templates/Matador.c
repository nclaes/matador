/*
 *
 * This application configures UART 16550 to baud rate 9600.
 * PS7 UART (Zynq) is not initialized by this application, since
 * bootrom/bsp configures it to baud rate 115200
 *
 * ------------------------------------------------
 * | UART TYPE   BAUD RATE                        |
 * ------------------------------------------------
 *   uartns550   9600
 *   uartlite    Configurable only in HW design
 *   ps7_uart    115200 (configured by bootrom/bsp)
 */
#include "instructions_and_data.h"

#define DEBUG 0

#if (!defined(DEBUG))
extern void xil_printf(const char *format, ...);
#endif

// #ifndef SDT
// int XAxiDma_Training(u16 DeviceId);
// #else
// int XAxiDma_Inference(UINTPTR BaseAddress);
// #endif

#ifndef SDT
int XAxiDma_Inference(u16 DeviceId);
#else
int XAxiDma_Inference(UINTPTR BaseAddress);
#endif

static int CheckData(int mode);


void print_matador(){

	printf("\n\r");
	printf("                                                                                                                        \n\r");
	printf("      #@@@@@#        &@@@@@,                  ,&&&/                               #@@@                                  \n\r");
	printf("      @@@@@@@@     ,@@@@@@@&                  /@@@                                @@@@,                                 \n\r");
	printf("      @@@@*@@@@    @@@@&@@@&   /@@@@@@@@@@*   /@@@@@@@    &@@@@@@@@@@     &@@@@@@@@@@@,   #@@@@@@@@@&    @@@@&@@@@@     \n\r");
	printf("      @@@@ *@@@(  @@@@ &@@@&   /((     &@@@#  /@@@(       ((/     @@@@  ,@@@@     @@@@,  @@@@     &@@@(  @@@@&          \n\r");
	printf("      @@@@  (@@@#@@@@  &@@@&   @@@@@@@@@@@@)  /@@@(      *@@@@@@@@@@@@  #@@@      @@@@, (@@@      ,@@@@  @@@@           \n\r");
	printf("      @@@@   #@@@@@@   &@@@&  @@@@     /@@@)  *@@@@     ,@@@@     @@@@  ,@@@@     @@@@,  @@@@     &@@@/  @@@@           \n\r");
	printf("      @@@@    #@@@@    &@@@&   @@@@@@@@@@@@)   *@@@@@@@# *@@@@@@@@@@@@    &@@@@@@@@@@@,   (@@@@@@@@@&    @@@@           \n\r");
	printf("                                                                                                                        \n\r");
	printf("\n\r");
}



/************************** Variable Definitions *****************************/
/*
 * Device instance definitions
 */
//The XAxiDma driver instance data. An instance must be allocated for each DMA engine in use.
XAxiDma AxiDma; 

/*****************************************************************************/
/**
* The entry point for this example. It invokes the example function,
* and reports the execution status.
*
* @param	None.
*
* @return
*		- XST_SUCCESS if example finishes successfully
*		- XST_FAILURE if example fails.
*
* @note		None.
*
******************************************************************************/
int main()
{
    init_platform();
    print_matador();
    xil_printf("%sMATADOR Acclerator\n\r", space);
    xil_printf("%sBare Metal Test	\n\r", space);
    
	int Status;

// 	/* Inference */
// #ifndef SDT
// //	xil_printf("%sSDT mode selected	\n\r", space);
// 	Status = XAxiDma_Training(DMA_DEV_ID);
// #else
// //	xil_printf("%sSDT mode not selected	\n\r", space);
// 	Status = XAxiDma_Training(XPAR_XAXIDMA_0_BASEADDR);
// #endif

#ifndef SDT
//	xil_printf("%sSDT mode selected	\n\r", space);
	Status = XAxiDma_Inference(DMA_DEV_ID);
#else
//	xil_printf("%sSDT mode not selected	\n\r", space);
	Status = XAxiDma_Inference(XPAR_XAXIDMA_0_BASEADDR);
#endif

    cleanup_platform();
    return XST_SUCCESS;
}

#if defined(XPAR_UARTNS550_0_BASEADDR)
/*****************************************************************************/
/*
*
* Uart16550 setup routine, need to set baudrate to 9600, and data bits to 8
*
* @param	None.
*
* @return	None
*
* @note		None.
*
******************************************************************************/
static void Uart550_Setup(void)
{

	/* Set the baudrate to be predictable
	 */
	XUartNs550_SetBaud(XPAR_UARTNS550_0_BASEADDR,
			   XPAR_XUARTNS550_CLOCK_HZ, 9600);

	XUartNs550_SetLineControlReg(XPAR_UARTNS550_0_BASEADDR,
				     XUN_LCR_8_DATA_BITS);

}
#endif

/*****************************************************************************/
/**
* The example to do the simple transfer through polling. The constant
* NUMBER_OF_TRANSFERS defines how many times a simple transfer is repeated.
*
* @param	DeviceId is the Device Id of the XAxiDma instance
*
* @return
*		- XST_SUCCESS if example finishes successfully
*		- XST_FAILURE if error occurs
*
* @note		None
*
*
******************************************************************************/
// #ifndef SDT
// int XAxiDma_Training(u16 DeviceId)
// #else
// int XAxiDma_Training(UINTPTR BaseAddress)
// #endif
// {
// 	XAxiDma_Config *CfgPtr;
// 	int Status;
// 	// int Tries = NUMBER_OF_TRANSFERS;
// 	int Index;
// 	// u64 *TxBufferPtr;
// 	unsigned long long  *RxBufferPtr;
// 	// u64 Value;
// 	int TimeOut = POLL_TIMEOUT_COUNTER;

// 	float freq = 0.00153846153;
// 	XTime tStart, tEnd;
// 	// double  ElapsedTime;

// 	int mode = 1; 
// 	// mode = 1 (Training)
// 	// mode = 0 

// 	// TxBufferPtr = (u64 *)TX_INSTRUCTION_BUFFER_BASE ;
// 	RxBufferPtr = (unsigned long long *)RX_TRAINING_DATA_BUFFER_BASE;

// 	/* Initialize the XAxiDma device.
// 	 */
// #ifndef SDT
// 	CfgPtr = XAxiDma_LookupConfig(DeviceId);
// 	if (!CfgPtr) {
// 		xil_printf("No config found for %d\r\n", DeviceId);
// 		return XST_FAILURE;
// 	}
// #else
// 	CfgPtr = XAxiDma_LookupConfig(BaseAddress);
// 	if (!CfgPtr) {
// 		xil_printf("No config found for %d\r\n", BaseAddress);
// 		return XST_FAILURE;
// 	}
// #endif

// 	Status = XAxiDma_CfgInitialize(&AxiDma, CfgPtr);
// 	if (Status != XST_SUCCESS) {
// 		xil_printf("Initialization failed %d\r\n", Status);
// 		return XST_FAILURE;
// 	}

// 	if (XAxiDma_HasSg(&AxiDma)) {
// 		xil_printf("Device configured as SG mode \r\n");
// 		return XST_FAILURE;
// 	}

// 	/* Disable interrupts, we use polling mode
// 	 */
// 	XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK,
// 			    XAXIDMA_DEVICE_TO_DMA);
// 	XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK,
// 			    XAXIDMA_DMA_TO_DEVICE);

// 	// Value = TEST_START_VALUE;

// 	// for (Index = 0; Index < NUMBER_OF_PKTS; Index ++) {
// 	// 	xil_printf("%sIndex: %llu\tSent: %llu\n\r", space, Index, TxDataBufferPtr[Index]);
// 	// }
// 	/* Flush the buffers before the DMA transfer, in case the Data Cache
// 	 * is enabled
// 	 */
// 	Xil_DCacheFlushRange((UINTPTR)Training_BufferPtr, MAX_TRAINING_PKT_LEN);
// 	Xil_DCacheFlushRange((UINTPTR)RxBufferPtr, MAX_TRAINING_DATAPOINT_LEN);

// 	for (Index = 0; Index < EPOCHS; Index ++) {

// 		XTime_GetTime(&tStart);

// 		Status = XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR) RxBufferPtr,
// 						MAX_TRAINING_DATAPOINT_LEN, XAXIDMA_DEVICE_TO_DMA);


// 		if (Status != XST_SUCCESS) {
// 			return XST_FAILURE;
// 		}

// 		Status = XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR) Training_BufferPtr,
// 						MAX_TRAINING_PKT_LEN, XAXIDMA_DMA_TO_DEVICE);

// 		if (Status != XST_SUCCESS) {
// 			return XST_FAILURE;
// 		}

// 		/*Wait till tranfer is done or 1usec * 10^6 iterations of timeout occurs*/
// 		while (TimeOut) {
// 			if (!(XAxiDma_Busy(&AxiDma, XAXIDMA_DEVICE_TO_DMA)) &&
// 			    !(XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE))) {
// 				break;
// 			}
// 			TimeOut--;
// 			usleep(1U);
// 		}

// 		XTime_GetTime(&tEnd);
// 		float clk_flt = (2*(tEnd - tStart))*(freq);
// 		xil_printf("%sEpoch [\t%d\t/%d]\n\r", space, Index+1, EPOCHS);
// 		xil_printf("%s[Training Mode]\033[1m Training \033[0m took \033[1m%llu\033[0m clock cycles.\n\r",space,  2*(tEnd - tStart));
// 		printf("%s[Training Mode]\033[1m Training \033[0m took \033[1m%f\033[0m us.\n\r",space, clk_flt);

// 		Status = CheckData(mode);
// 		if (Status != XST_SUCCESS) {
// 			return XST_FAILURE;
// 		}
// 	}
// 	return XST_SUCCESS;
// }

#ifndef SDT
int XAxiDma_Inference(u16 DeviceId)
#else
int XAxiDma_Inference(UINTPTR BaseAddress)
#endif
{
	XAxiDma_Config *CfgPtr;
	int Status;
	// int Tries = NUMBER_OF_TRANSFERS;
	int Index;
	// u64 *TxBufferPtr;
	unsigned long long  *RxBufferPtr;
	// u64 Value;
	int TimeOut = POLL_TIMEOUT_COUNTER;

	float freq = 0.00153846153;
	XTime tStart_, tEnd_;
	// double  ElapsedTime;

	int mode = 0; 
	// mode = 1 (Training)
	// mode = 0 

	// TxBufferPtr = (u64 *)TX_INSTRUCTION_BUFFER_BASE ;
	RxBufferPtr = (unsigned long long *)RX_TESTING_DATA_BUFFER_BASE;

	/* Initialize the XAxiDma device.
	 */
#ifndef SDT
	CfgPtr = XAxiDma_LookupConfig(DeviceId);
	if (!CfgPtr) {
		xil_printf("No config found for %d\r\n", DeviceId);
		return XST_FAILURE;
	}
#else
	CfgPtr = XAxiDma_LookupConfig(BaseAddress);
	if (!CfgPtr) {
		xil_printf("No config found for %d\r\n", BaseAddress);
		return XST_FAILURE;
	}
#endif

	Status = XAxiDma_CfgInitialize(&AxiDma, CfgPtr);
	if (Status != XST_SUCCESS) {
		xil_printf("Initialization failed %d\r\n", Status);
		return XST_FAILURE;
	}

	if (XAxiDma_HasSg(&AxiDma)) {
		xil_printf("Device configured as SG mode \r\n");
		return XST_FAILURE;
	}

	/* Disable interrupts, we use polling mode
	 */
	XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK,
			    XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK,
			    XAXIDMA_DMA_TO_DEVICE);

	// Value = TEST_START_VALUE;

	// for (Index = 0; Index < NUMBER_OF_PKTS; Index ++) {
	// 	xil_printf("%sIndex: %llu\tSent: %llu\n\r", space, Index, TxDataBufferPtr[Index]);
	// }
	/* Flush the buffers before the DMA transfer, in case the Data Cache
	 * is enabled
	 */
	Xil_DCacheFlushRange((UINTPTR)Inference_BufferPtr, MAX_TESTING_PKT_LEN);
	Xil_DCacheFlushRange((UINTPTR)RxBufferPtr, MAX_TESTING_DATAPOINT_LEN);

	for (Index = 0; Index < 1; Index ++) {

		XTime_GetTime(&tStart_);

		Status = XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR) RxBufferPtr,
						MAX_TESTING_DATAPOINT_LEN, XAXIDMA_DEVICE_TO_DMA);


		if (Status != XST_SUCCESS) {
			return XST_FAILURE;
		}

		Status = XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR) Inference_BufferPtr,
						MAX_TESTING_PKT_LEN, XAXIDMA_DMA_TO_DEVICE);

		if (Status != XST_SUCCESS) {
			return XST_FAILURE;
		}

		/*Wait till tranfer is done or 1usec * 10^6 iterations of timeout occurs*/
		while (TimeOut) {
			if (!(XAxiDma_Busy(&AxiDma, XAXIDMA_DEVICE_TO_DMA)) &&
			    !(XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE))) {
				break;
			}
			TimeOut--;
			usleep(1U);
		}

		XTime_GetTime(&tEnd_);
		float clk_flt_ = 0.0;
		xil_printf("%s\n\r", space);
		xil_printf("%s[Testing Mode]\033[1m Inference \033[0m took \033[1m%llu\033[0m clock cycles.\n\r",space,  2*(tEnd_ - tStart_));
		clk_flt_ = (2*(tEnd_ - tStart_))*(freq);
		printf("%s[Testing Mode]\033[1m Inference \033[0m took \033[1m%f\033[0m us.\n\r",space, clk_flt_);

		Status = CheckData(mode);
		if (Status != XST_SUCCESS) {
			return XST_FAILURE;
		}
	}
	return XST_SUCCESS;
}

static int CheckData(int mode)
{
	unsigned long long *RxPacket;
	unsigned long long *RxPacket_test;
	int Index = 0;
	float error = 0.0; 
	u32 Value;
	// u32 Class; 
	// u32 Pred; 
	float acc = 0.0; 

	if(mode){
		// RxPacket = (unsigned long long *) RX_TRAINING_DATA_BUFFER_BASE;

		// Xil_DCacheInvalidateRange((UINTPTR)RxPacket, MAX_TRAINING_DATAPOINT_LEN);

		// for (Index = 0; Index <NUMBER_Of_TRAINING_DATAPOINTS; Index++) {
		// 	Value = RxPacket[Index];
		// 	Class = Value & 0x000F;
		// 	Pred  = Value & 0x00F0; 
		// 	Pred  = Pred >> 4;  
		// 	if(Class != Pred){
		// 		error += 1; 
		// 	}
		// }
		// acc = 1.0 - (error/NUMBER_Of_TRAINING_DATAPOINTS);

		// printf("%s[Training Mode] Training Acc: %f\n\r", space, acc);
	}
	else{
		RxPacket_test = (unsigned long long *) RX_TESTING_DATA_BUFFER_BASE;

		Xil_DCacheInvalidateRange((UINTPTR)RxPacket_test, MAX_TESTING_DATAPOINT_LEN);

		for (Index = 0; Index <10; Index++) {
			Value = RxPacket_test[Index];
			// Class = Value & 0x000F;
			// Pred  = Value & 0x00F0; 
			// Pred  = Pred >> 4;  
			xil_printf("%s[%d] Predicted: %d\n\r", space, Index, Value);

		}
		
		// acc = 1.0 - (error/NUMBER_Of_TESTING_DATAPOINTS);
		// printf("%s[Testing Mode] Test Acc: %f\n\r", space, acc);
	}

	return XST_SUCCESS;
}
