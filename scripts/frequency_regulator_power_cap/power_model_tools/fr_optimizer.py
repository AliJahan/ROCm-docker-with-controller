import z3 

avg_power = 100 # watts 
power_price = 10 # $/watt
max_power = 150
reduce_price = 1 #$/watt 
min_acceptable_power = 50 # watt (based on throughput, decided later)
increase_price = 10 #$/watt

def calculate_items(new_power, reduce_amount, increase_amount, avg_power, power_price, reduce_price, increase_price):
    baseline_cost = avg_power*power_price
    fr_cost = new_power*power_price - reduce_price*reduce_amount - increase_price*increase_amount
    saving = baseline_cost - fr_cost
    # s = z3.Solver()
    # s.add(baseline_cost)
    return {
        "costs": {
            "baseline_elec_cost": baseline_cost,
            "fr_elec_cost": new_power*power_price,
            'reg_up_reward': reduce_amount * reduce_price,
            'reg_down_reward': increase_amount * increase_price,
            'fr_total_cost': fr_cost,
            'saving': saving
        },
        "regulation": {
            'reg_up': reduce_amount,
            'reg_down': increase_amount
        },
        "powers": {
            'baseline_power': avg_power,
            'offset_power': new_power - avg_power,
            'fr_power': new_power
        }
    }

def Optimize(
        avg_power = 100, 
        power_price = 10, 
        max_power = 150, 
        min_acceptable_power = 50, 
        reduce_price = 6, 
        increase_price = 2, 
        saving_threashold_pct = 20,
        symmetric_provision_range = False,
        predicted_perf_score_pct=90
    ):
    # Decision variables
    avg_power = round(avg_power)
    power_price = round(power_price)
    max_power = round(max_power)
    min_acceptable_power = round(min_acceptable_power)
    reduce_price = round(reduce_price)
    increase_price = round(increase_price)
    # print(f"optimize(avg_power = {avg_power}, power_price = {power_price}, max_power = {max_power}, min_acceptable_power = {min_acceptable_power}, reduce_price = {reduce_price}, increase_price = {increase_price}, saving_threashold_pct = {saving_threashold_pct})")
    
    try:
        assert avg_power > 0
        assert power_price > 0
        assert max_power > 0
        assert min_acceptable_power > 0
        assert reduce_price > 0
        assert increase_price > 0
        assert power_price > 0
    except:
        return None

    
    reduce_amount = z3.Int('reduce_amount')
    increase_amount = z3.Int('increase_amount')
    new_power = z3.Int('new_power')

    # Constraints
    c1 = z3.And(0 <= reduce_amount, reduce_amount <= new_power - int(min_acceptable_power)) # 10% of avg power
    # c2 = z3.And(0 <= increase_amount, increase_amount <= (new_power - int(avg_power))) 
    c3 = z3.And(0 <= increase_amount, increase_amount <= (int(max_power) - new_power)) 

    c4 = z3.And(int(avg_power) <= new_power, new_power <= int(max_power))
    # c4 = z3.And(avg_power + increase_amount <= max_power, avg_power - increase_amount >= min_power)
    # Objective function
    baseline_cost = int(avg_power) * int(power_price)
    fr_cost = new_power*int(power_price) - (int(reduce_price)*reduce_amount + int(increase_price)*increase_amount)*predicted_perf_score_pct/100
    saving = baseline_cost - fr_cost
    saving_threashold_dollar = int(baseline_cost * (100 - saving_threashold_pct) / 100.0)
    # c4 = z3.Bool(saving_threashold_dollar <= saving)
    # c2 = z3.And(fr_cost <= new_power, new_power <= max_power)
    # print(f"@{saving_threashold_dollar}")
    # Optimization problem
    opt = z3.Optimize()
    opt.maximize(saving)
    opt.add(c1)
    if symmetric_provision_range:
        opt.add(increase_amount == reduce_amount)
    opt.add(c3)
    opt.add(c4)
    opt.add(saving_threashold_dollar >= fr_cost)
    opt.add(new_power + increase_amount <= int(max_power))

    if opt.check() == z3.sat:
        m = opt.model()
        # print(m)
        _new_power = m[new_power].as_long()
        _reduce_amount = m[reduce_amount].as_long()
        _increase_amount = m[increase_amount].as_long()
        return calculate_items(_new_power, _reduce_amount, _increase_amount, avg_power, power_price, reduce_price, increase_price)
        # print("Minimized cost:", m[cost])
        # print("Reduce amount:", m[reduce_amount]) 
        # print("Increase amount:", m[increase_amount])
    else:
        # print(f"optimize(avg_power = {avg_power}, power_price = {power_price}, max_power = {max_power}, min_acceptable_power = {min_acceptable_power}, reduce_price = {reduce_price}, increase_price = {increase_price}, saving_threashold_pct = {saving_threashold_pct})")
        return None


if __name__ == "__main__":
    for i in range(10, 100, 10):
        print(f"price for saving threashold {i}:", end=" ")
        print(Optimize(saving_threashold_pct=i))