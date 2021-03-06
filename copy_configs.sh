#!/bin/sh


#pr_page_fault_pinned_second_iteration_offset_P100 pr_cache_prefetch_test_managed_P100 pr_overhead_P100 pr_tlb_miss_second_iteration_offset_P100 pr_page_fault_preferredlocation_second_iteration_P100 pr_cache_flush_test_managed_P100 pr_page_fault_accessby_first_iteration_P100 pr_tlb_miss_second_iteration_P100 pr_tlb_miss_second_iteration_P100_disable_L1 pr_cache_prefetch_test_P100 pr_page_fault_pinned_second_iteration_offset_P100_disable_L1 pr_L1_size_P100 pr_page_fault_preferredlocation_first_iteration_P100 pr_page_fault_plain_managed_first_iteration_P100 pr_L2_line_size_P100_disable_L1 pr_page_fault_accessby_second_iteration_P100 pr_L2_size_P100 pr_tlb_miss_first_iteration_P100 pr_page_fault_pinned_second_iteration_P100 pr_page_fault_plain_managed_second_iteration_P100 pr_tlb_miss_second_iteration_offset_P100_disable_L1 pr_cache_flush_test_P100 pr_L1_line_size_P100 pr_page_fault_pinned_second_iteration_P100_disable_L1 pr_page_fault_pinned_first_iteration_P100

for configs in pr_page_fault_pinned_second_iteration_offset_K40 pr_cache_prefetch_test_managed_K40 pr_overhead_K40 pr_tlb_miss_second_iteration_offset_K40 pr_page_fault_preferredlocation_second_iteration_K40 pr_cache_flush_test_managed_K40 pr_page_fault_accessby_first_iteration_K40 pr_tlb_miss_second_iteration_K40 pr_tlb_miss_second_iteration_K40_disable_L1 pr_cache_prefetch_test_K40 pr_page_fault_pinned_second_iteration_offset_K40_disable_L1 pr_L1_size_K40 pr_page_fault_preferredlocation_first_iteration_K40 pr_page_fault_plain_managed_first_iteration_K40 pr_L2_line_size_K40_disable_L1 pr_page_fault_accessby_second_iteration_K40 pr_L2_size_K40 pr_tlb_miss_first_iteration_K40 pr_page_fault_pinned_second_iteration_K40 pr_page_fault_plain_managed_second_iteration_K40 pr_tlb_miss_second_iteration_offset_K40_disable_L1 pr_cache_flush_test_K40 pr_L1_line_size_K40 pr_page_fault_pinned_second_iteration_K40_disable_L1 pr_page_fault_pinned_first_iteration_K40

do
new_config=$(echo $configs | sed -e "s/K40/V100/g")
mkdir $new_config
cd $new_config
cp ../$configs/Makefile Makefile
cp ../$configs/$configs.cu $new_config.cu
cd ..
done
